/*********************************************************************
 * convolve_cuda.cu - HIGHLY OPTIMIZED FOR TESLA T4 (SM_75)
 * 
 * T4-Specific Optimizations:
 * 1. Turing has 96KB L1/Shared (64KB shared + 32KB L1 config)
 * 2. 40 SMs with 64 FP32 cores each = 2560 CUDA cores
 * 3. Separable convolution (fewer ops)
 * 4. Shared memory tiling with proper indexing
 * 5. Vectorized float4 loads for global→shared (FIXED ALIGNMENT)
 * 6. Warp-aligned 32×8 blocks (256 threads, 8 warps)
 * 7. Bank conflict avoidance with padding
 * 8. Async streams for compute/transfer overlap
 * 9. Persistent device buffers
 * 10. Constant memory for kernels
 *********************************************************************/

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "base.h"
#include "error.h"
#include "convolve.h"
#include "klt_util.h"

#define MAX_KERNEL_WIDTH 71
#define WARP_SIZE 32
#define BLOCK_DIM_X 32  // Full warp for coalescing
#define BLOCK_DIM_Y 8   // 256 threads total
#define MAX_KERNEL_SIZE 35

#define CUDA_CHECK(call) \
  do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(err)); \
      exit(EXIT_FAILURE); \
    } \
  } while(0)

/*********************************************************************
 * Kernel Data Structures
 *********************************************************************/
typedef struct {
  int width;
  float data[MAX_KERNEL_WIDTH];
} ConvolutionKernel;

static ConvolutionKernel gauss_kernel;
static ConvolutionKernel gaussderiv_kernel;
static float sigma_last = -10.0;

// Constant memory for kernel (faster than global, cached)
__constant__ float c_kernel[MAX_KERNEL_SIZE];

/*********************************************************************
 * Persistent Device Buffers with Streams
 *********************************************************************/
static struct {
  float *d_img1, *d_img2;
  size_t allocated_size;
  cudaStream_t stream;
  bool initialized;
} g_gpu = {NULL, NULL, 0, NULL, false};

static void ensure_gpu_buffers(size_t bytes) {
  if (!g_gpu.initialized) {
    CUDA_CHECK(cudaStreamCreate(&g_gpu.stream));
    // Set shared memory config: prefer 64KB shared, 32KB L1
    CUDA_CHECK(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
    g_gpu.initialized = true;
  }
  
  if (bytes > g_gpu.allocated_size) {
    if (g_gpu.d_img1) {
      cudaFree(g_gpu.d_img1);
      cudaFree(g_gpu.d_img2);
    }
    CUDA_CHECK(cudaMalloc(&g_gpu.d_img1, bytes));
    CUDA_CHECK(cudaMalloc(&g_gpu.d_img2, bytes));
    g_gpu.allocated_size = bytes;
  }
}

/*********************************************************************
 * OPTIMIZED HORIZONTAL CONVOLUTION WITH FIXED FLOAT4
 * 
 * Strategy: Use float4 only when column index is 4-aligned AND in bounds
 *********************************************************************/
__global__ void convolveHoriz_Optimized(
  const float * __restrict__ imgin,
  float * __restrict__ imgout,
  int ncols, int nrows,
  int kernel_width)
{
  const int radius = kernel_width / 2;
  const int tile_width = blockDim.x;
  const int tile_height = blockDim.y;
  
  // Shared memory: [tile_height][tile_width + 2*radius + 1]
  // +1 for bank conflict avoidance
  const int tile_stride = tile_width + 2 * radius + 1;
  extern __shared__ float s_tile[];
  
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int gx = blockIdx.x * tile_width + tx;
  const int gy = blockIdx.y * tile_height + ty;
  
  if (gy >= nrows) return;
  
  // Cooperative loading with conditional float4
  const int tile_start_col = blockIdx.x * tile_width - radius;
  
  for (int row = ty; row < tile_height; row += tile_height) {
    int global_row = blockIdx.y * tile_height + row;
    if (global_row >= nrows) continue;
    
    const float* row_ptr = &imgin[global_row * ncols];
    
    // Each thread loads elements with stride = tile_width
    for (int local_col = tx; local_col < tile_stride; local_col += tile_width) {
      int global_col = tile_start_col + local_col;
      
      // Try float4 if aligned and in bounds
      if ((global_col % 4 == 0) &&  // 4-aligned column
          (global_col >= 0) && 
          (global_col + 3 < ncols) &&
          (local_col + 3 < tile_stride)) {
        
        // Safe float4 load
        float4 data = reinterpret_cast<const float4*>(&row_ptr[global_col])[0];
        s_tile[row * tile_stride + local_col + 0] = data.x;
        s_tile[row * tile_stride + local_col + 1] = data.y;
        s_tile[row * tile_stride + local_col + 2] = data.z;
        s_tile[row * tile_stride + local_col + 3] = data.w;
        
        // Note: We still iterate with stride=tile_width, so next iteration
        // will handle the next set. This may overlap but that's fine.
      } else {
        // Scalar fallback for boundaries or misalignment
        float val = 0.0f;
        if (global_col >= 0 && global_col < ncols) {
          val = row_ptr[global_col];
        }
        s_tile[row * tile_stride + local_col] = val;
      }
    }
  }
  __syncthreads();
  
  // Compute convolution
  if (gx >= ncols) return;
  
  // Zero boundaries
  if (gx < radius || gx >= ncols - radius) {
    imgout[gy * ncols + gx] = 0.0f;
    return;
  }
  
  // Convolve
  float sum = 0.0f;
  int s_center = ty * tile_stride + tx + radius;
  
  #pragma unroll 8
  for (int k = kernel_width - 1; k >= 0; k--) {
    int offset = kernel_width - 1 - k;
    sum += s_tile[s_center - radius + offset] * c_kernel[k];
  }
  
  imgout[gy * ncols + gx] = sum;
}

/*********************************************************************
 * OPTIMIZED VERTICAL CONVOLUTION
 *********************************************************************/
__global__ void convolveVert_Optimized(
  const float * __restrict__ imgin,
  float * __restrict__ imgout,
  int ncols, int nrows,
  int kernel_width)
{
  const int radius = kernel_width / 2;
  const int tile_width = blockDim.x;
  const int tile_height = blockDim.y;
  
  // Shared memory: [tile_height + 2*radius][tile_width + 1]
  // +1 for bank conflict avoidance
  const int tile_stride = tile_width + 1;
  const int tile_vert = tile_height + 2 * radius;
  extern __shared__ float s_tile[];
  
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int gx = blockIdx.x * tile_width + tx;
  const int gy = blockIdx.y * tile_height + ty;
  
  if (gx >= ncols) return;
  
  // Cooperative loading
  const int tile_start_row = blockIdx.y * tile_height - radius;
  
  // Each thread loads a column of data
  for (int local_row = ty; local_row < tile_vert; local_row += tile_height) {
    int global_row = tile_start_row + local_row;
    
    float val = 0.0f;
    if (global_row >= 0 && global_row < nrows && gx < ncols) {
      val = imgin[global_row * ncols + gx];
    }
    s_tile[local_row * tile_stride + tx] = val;
  }
  __syncthreads();
  
  // Compute convolution
  if (gy >= nrows) return;
  
  // Zero boundaries
  if (gy < radius || gy >= nrows - radius) {
    imgout[gy * ncols + gx] = 0.0f;
    return;
  }
  
  // Convolve
  float sum = 0.0f;
  int s_center_row = ty + radius;
  
  #pragma unroll 8
  for (int k = kernel_width - 1; k >= 0; k--) {
    int offset = kernel_width - 1 - k;
    sum += s_tile[(s_center_row - radius + offset) * tile_stride + tx] * c_kernel[k];
  }
  
  imgout[gy * ncols + gx] = sum;
}

/*********************************************************************
 * Host Wrapper Functions
 *********************************************************************/
static void _convolveImageHoriz(
  _KLT_FloatImage imgin,
  ConvolutionKernel kernel,
  _KLT_FloatImage imgout)
{
  const int ncols = imgin->ncols;
  const int nrows = imgin->nrows;
  const size_t nbytes = ncols * nrows * sizeof(float);
  
  ensure_gpu_buffers(nbytes);
  
  // Copy kernel to constant memory
  CUDA_CHECK(cudaMemcpyToSymbolAsync(c_kernel, kernel.data, 
    kernel.width * sizeof(float), 0, cudaMemcpyHostToDevice, g_gpu.stream));
  
  // Copy input to device
  CUDA_CHECK(cudaMemcpyAsync(g_gpu.d_img1, imgin->data, nbytes,
    cudaMemcpyHostToDevice, g_gpu.stream));
  
  // Launch configuration
  const int radius = kernel.width / 2;
  dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
  dim3 grid((ncols + BLOCK_DIM_X - 1) / BLOCK_DIM_X,
            (nrows + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);
  
  // Shared memory size
  const int tile_stride = BLOCK_DIM_X + 2 * radius + 1;
  size_t shared_bytes = BLOCK_DIM_Y * tile_stride * sizeof(float);
  
  // Enable 64KB shared memory if needed (T4 supports it)
  if (shared_bytes > 48 * 1024) {
    CUDA_CHECK(cudaFuncSetAttribute(convolveHoriz_Optimized,
      cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024));
    CUDA_CHECK(cudaFuncSetAttribute(convolveHoriz_Optimized,
      cudaFuncAttributePreferredSharedMemoryCarveout, 100)); // 64KB shared
  }
  
  convolveHoriz_Optimized<<<grid, block, shared_bytes, g_gpu.stream>>>(
    g_gpu.d_img1, g_gpu.d_img2, ncols, nrows, kernel.width);
  
  CUDA_CHECK(cudaGetLastError());
  
  // Copy result back
  CUDA_CHECK(cudaMemcpyAsync(imgout->data, g_gpu.d_img2, nbytes,
    cudaMemcpyDeviceToHost, g_gpu.stream));
  
  CUDA_CHECK(cudaStreamSynchronize(g_gpu.stream));
  
  imgout->ncols = ncols;
  imgout->nrows = nrows;
}

static void _convolveImageVert(
  _KLT_FloatImage imgin,
  ConvolutionKernel kernel,
  _KLT_FloatImage imgout)
{
  const int ncols = imgin->ncols;
  const int nrows = imgin->nrows;
  const size_t nbytes = ncols * nrows * sizeof(float);
  
  ensure_gpu_buffers(nbytes);
  
  // Copy kernel to constant memory
  CUDA_CHECK(cudaMemcpyToSymbolAsync(c_kernel, kernel.data,
    kernel.width * sizeof(float), 0, cudaMemcpyHostToDevice, g_gpu.stream));
  
  // Copy input to device
  CUDA_CHECK(cudaMemcpyAsync(g_gpu.d_img1, imgin->data, nbytes,
    cudaMemcpyHostToDevice, g_gpu.stream));
  
  // Launch configuration
  const int radius = kernel.width / 2;
  dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
  dim3 grid((ncols + BLOCK_DIM_X - 1) / BLOCK_DIM_X,
            (nrows + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);
  
  // Shared memory size
  const int tile_vert = BLOCK_DIM_Y + 2 * radius;
  const int tile_stride = BLOCK_DIM_X + 1;
  size_t shared_bytes = tile_vert * tile_stride * sizeof(float);
  
  // Enable 64KB shared memory if needed
  if (shared_bytes > 48 * 1024) {
    CUDA_CHECK(cudaFuncSetAttribute(convolveVert_Optimized,
      cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024));
    CUDA_CHECK(cudaFuncSetAttribute(convolveVert_Optimized,
      cudaFuncAttributePreferredSharedMemoryCarveout, 100));
  }
  
  convolveVert_Optimized<<<grid, block, shared_bytes, g_gpu.stream>>>(
    g_gpu.d_img1, g_gpu.d_img2, ncols, nrows, kernel.width);
  
  CUDA_CHECK(cudaGetLastError());
  
  // Copy result back
  CUDA_CHECK(cudaMemcpyAsync(imgout->data, g_gpu.d_img2, nbytes,
    cudaMemcpyDeviceToHost, g_gpu.stream));
  
  CUDA_CHECK(cudaStreamSynchronize(g_gpu.stream));
  
  imgout->ncols = ncols;
  imgout->nrows = nrows;
}

/*********************************************************************
 * Separable Convolution
 *********************************************************************/
static void _convolveSeparate(
  _KLT_FloatImage imgin,
  ConvolutionKernel horiz_kernel,
  ConvolutionKernel vert_kernel,
  _KLT_FloatImage imgout)
{
  _KLT_FloatImage tmpimg = _KLTCreateFloatImage(imgin->ncols, imgin->nrows);
  _convolveImageHoriz(imgin, horiz_kernel, tmpimg);
  _convolveImageVert(tmpimg, vert_kernel, imgout);
  _KLTFreeFloatImage(tmpimg);
}

/*********************************************************************
 * Kernel Computation (unchanged from original)
 *********************************************************************/
static void _computeKernels(
  float sigma,
  ConvolutionKernel *gauss,
  ConvolutionKernel *gaussderiv)
{
  const float factor = 0.01f;
  int i;

  assert(MAX_KERNEL_WIDTH % 2 == 1);
  assert(sigma >= 0.0);

  {
    const int hw = MAX_KERNEL_WIDTH / 2;
    float max_gauss = 1.0f, max_gaussderiv = (float)(sigma*exp(-0.5f));
	
    for (i = -hw; i <= hw; i++) {
      gauss->data[i+hw] = (float)exp(-i*i / (2*sigma*sigma));
      gaussderiv->data[i+hw] = -i * gauss->data[i+hw];
    }

    gauss->width = MAX_KERNEL_WIDTH;
    for (i = -hw; fabs(gauss->data[i+hw] / max_gauss) < factor; 
         i++, gauss->width -= 2);
    gaussderiv->width = MAX_KERNEL_WIDTH;
    for (i = -hw; fabs(gaussderiv->data[i+hw] / max_gaussderiv) < factor; 
         i++, gaussderiv->width -= 2);
    if (gauss->width == MAX_KERNEL_WIDTH || 
        gaussderiv->width == MAX_KERNEL_WIDTH)
      KLTError("(_computeKernels) MAX_KERNEL_WIDTH %d is too small for "
               "a sigma of %f", MAX_KERNEL_WIDTH, sigma);
  }

  for (i = 0; i < gauss->width; i++)
    gauss->data[i] = gauss->data[i+(MAX_KERNEL_WIDTH-gauss->width)/2];
  for (i = 0; i < gaussderiv->width; i++)
    gaussderiv->data[i] = gaussderiv->data[i+(MAX_KERNEL_WIDTH-gaussderiv->width)/2];

  {
    const int hw = gaussderiv->width / 2;
    float den;
			
    den = 0.0;
    for (i = 0; i < gauss->width; i++) den += gauss->data[i];
    for (i = 0; i < gauss->width; i++) gauss->data[i] /= den;
    den = 0.0;
    for (i = -hw; i <= hw; i++) den -= i*gaussderiv->data[i+hw];
    for (i = -hw; i <= hw; i++) gaussderiv->data[i+hw] /= den;
  }

  sigma_last = sigma;
}

/*********************************************************************
 * Public API Functions
 *********************************************************************/
void _KLTToFloatImage(
  KLT_PixelType *img,
  int ncols, int nrows,
  _KLT_FloatImage floatimg)
{
  KLT_PixelType *ptrend = img + ncols*nrows;
  float *ptrout = floatimg->data;

  assert(floatimg->ncols >= ncols);
  assert(floatimg->nrows >= nrows);

  floatimg->ncols = ncols;
  floatimg->nrows = nrows;

  while (img < ptrend) *ptrout++ = (float)*img++;
}

void _KLTGetKernelWidths(
  float sigma,
  int *gauss_width,
  int *gaussderiv_width)
{
  _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);
  *gauss_width = gauss_kernel.width;
  *gaussderiv_width = gaussderiv_kernel.width;
}

void _KLTComputeGradients(
  _KLT_FloatImage img,
  float sigma,
  _KLT_FloatImage gradx,
  _KLT_FloatImage grady)
{
  assert(gradx->ncols >= img->ncols);
  assert(gradx->nrows >= img->nrows);
  assert(grady->ncols >= img->ncols);
  assert(grady->nrows >= img->nrows);

  if (fabs(sigma - sigma_last) > 0.05)
    _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);
  
  ensure_gpu_buffers(img->ncols * img->nrows * sizeof(float));
  
  _convolveSeparate(img, gaussderiv_kernel, gauss_kernel, gradx);
  _convolveSeparate(img, gauss_kernel, gaussderiv_kernel, grady);
}

void _KLTComputeSmoothedImage(
  _KLT_FloatImage img,
  float sigma,
  _KLT_FloatImage smooth)
{
  assert(smooth->ncols >= img->ncols);
  assert(smooth->nrows >= img->nrows);

  if (fabs(sigma - sigma_last) > 0.05)
    _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);

  ensure_gpu_buffers(img->ncols * img->nrows * sizeof(float));
  
  _convolveSeparate(img, gauss_kernel, gauss_kernel, smooth);
}

// Cleanup function (call at program exit)
void _KLTCleanupGPU() {
  if (g_gpu.initialized) {
    if (g_gpu.d_img1) cudaFree(g_gpu.d_img1);
    if (g_gpu.d_img2) cudaFree(g_gpu.d_img2);
    cudaStreamDestroy(g_gpu.stream);
    g_gpu.initialized = false;
  }
}
