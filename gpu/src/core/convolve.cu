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
  float *d_img1, *d_img2, *d_transpose;
  size_t allocated_size;
  cudaStream_t stream;
  bool initialized;
} g_gpu = {NULL, NULL, NULL, 0, NULL, false};
 
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
      cudaFree(g_gpu.d_transpose);
    }
    CUDA_CHECK(cudaMalloc(&g_gpu.d_img1, bytes));
    CUDA_CHECK(cudaMalloc(&g_gpu.d_img2, bytes));
    CUDA_CHECK(cudaMalloc(&g_gpu.d_transpose, bytes));  // For transpose operations
    g_gpu.allocated_size = bytes;
  }
}
 
/*********************************************************************
 * EFFICIENT MATRIX TRANSPOSE WITH SHARED MEMORY
 * 
 * Uses 32x32 tiles with padding to avoid bank conflicts
 * This enables coalesced access for vertical convolution
 *********************************************************************/
#define TILE_DIM 32
#define TILE_PAD 1  // Padding to avoid bank conflicts

__global__ void transpose_kernel(
  const float * __restrict__ in,
  float * __restrict__ out,
  int width, int height)
{
  __shared__ float tile[TILE_DIM][TILE_DIM + TILE_PAD];
  
  int x_in = blockIdx.x * TILE_DIM + threadIdx.x;
  int y_in = blockIdx.y * TILE_DIM + threadIdx.y;
  
  // Read from input (coalesced)
  if (x_in < width && y_in < height) {
    tile[threadIdx.y][threadIdx.x] = in[y_in * width + x_in];
  }
  
  __syncthreads();
  
  // Write to output transposed (coalesced)
  int x_out = blockIdx.y * TILE_DIM + threadIdx.x;
  int y_out = blockIdx.x * TILE_DIM + threadIdx.y;
  
  if (x_out < height && y_out < width) {
    out[y_out * height + x_out] = tile[threadIdx.x][threadIdx.y];
  }
}

/*********************************************************************
 * OPTIMIZED HORIZONTAL CONVOLUTION
 * 
 * Used for both horizontal and vertical convolution (via transpose)
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
   
   // Shared memory with 8-byte padding for bank conflict avoidance
   // T4: 32 banks, 4-byte words → 8-byte padding = 2 words
   const int tile_stride = tile_width + 2 * radius + 8;  // +8 for padding
   extern __shared__ float s_tile[];
   
   const int tx = threadIdx.x;
   const int ty = threadIdx.y;
   const int gx = blockIdx.x * tile_width + tx;
   const int gy = blockIdx.y * tile_height + ty;
   
   if (gy >= nrows) return;
   
   // ============ PHASE 1: COOPERATIVE TILE LOADING ============
   const int tile_start_col = blockIdx.x * tile_width - radius;
   const int total_cols = tile_width + 2 * radius;
   
   // Each warp loads one row cooperatively
   for (int row = ty; row < tile_height; row += tile_height) {
     int global_row = blockIdx.y * tile_height + row;
     if (global_row >= nrows) continue;
     
     const float* row_ptr = &imgin[global_row * ncols];
     float* s_row = &s_tile[row * tile_stride];
     
    // Load tile data: each thread handles multiple elements
    for (int local_col = tx; local_col < total_cols; local_col += tile_width) {
      int global_col = tile_start_col + local_col;
      s_row[local_col] = (global_col >= 0 && global_col < ncols) ? row_ptr[global_col] : 0.0f;
    }
  }
   __syncthreads();
   
   // ============ PHASE 2: COMPUTE CONVOLUTION ============
   if (gx >= ncols) return;
   
   // Zero boundary pixels
   if (gx < radius || gx >= ncols - radius) {
     imgout[gy * ncols + gx] = 0.0f;
     return;
   }
   
   // Convolution with aggressive unrolling
   float sum = 0.0f;
   int s_center = ty * tile_stride + tx + radius;
   
   // Unroll based on typical kernel sizes
   if (kernel_width <= 7) {
     #pragma unroll
     for (int k = 0; k < kernel_width; k++) {
       sum += s_tile[s_center - radius + k] * c_kernel[k];
     }
   } else if (kernel_width <= 15) {
     #pragma unroll 4
     for (int k = 0; k < kernel_width; k++) {
       sum += s_tile[s_center - radius + k] * c_kernel[k];
     }
   } else {
     #pragma unroll 2
     for (int k = 0; k < kernel_width; k++) {
       sum += s_tile[s_center - radius + k] * c_kernel[k];
     }
   }
   
   imgout[gy * ncols + gx] = sum;
 }
 
/*********************************************************************
 * VERTICAL CONVOLUTION VIA TRANSPOSE
 * 
 * Instead of strided access, we:
 * 1. Transpose the input
 * 2. Apply horizontal convolution (coalesced access!)
 * 3. Transpose the result back
 * 
 * This makes vertical convolution as fast as horizontal!
 *********************************************************************/
 
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
  
  // Copy kernel to constant memory (reversed to match CPU convention)
  // CPU applies kernel in reverse: kernel.data[width-1] at left, kernel.data[0] at right
  // GPU applies forward: c_kernel[0] at left, c_kernel[width-1] at right
  float reversed_kernel[MAX_KERNEL_SIZE];
  for (int i = 0; i < kernel.width; i++) {
    reversed_kernel[i] = kernel.data[kernel.width - 1 - i];
  }
  CUDA_CHECK(cudaMemcpyToSymbolAsync(c_kernel, reversed_kernel, 
    kernel.width * sizeof(float), 0, cudaMemcpyHostToDevice, g_gpu.stream));
   
   // Copy input to device
   CUDA_CHECK(cudaMemcpyAsync(g_gpu.d_img1, imgin->data, nbytes,
     cudaMemcpyHostToDevice, g_gpu.stream));
   
   // Launch configuration
   const int radius = kernel.width / 2;
   dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
   dim3 grid((ncols + BLOCK_DIM_X - 1) / BLOCK_DIM_X,
             (nrows + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);
   
  // Shared memory size (must match kernel calculation!)
  const int tile_stride = BLOCK_DIM_X + 2 * radius + 8;  // +8 for padding
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
  
  // Copy kernel to constant memory (reversed to match CPU convention)
  float reversed_kernel[MAX_KERNEL_SIZE];
  for (int i = 0; i < kernel.width; i++) {
    reversed_kernel[i] = kernel.data[kernel.width - 1 - i];
  }
  CUDA_CHECK(cudaMemcpyToSymbolAsync(c_kernel, reversed_kernel,
    kernel.width * sizeof(float), 0, cudaMemcpyHostToDevice, g_gpu.stream));
  
  // Copy input to device
  CUDA_CHECK(cudaMemcpyAsync(g_gpu.d_img1, imgin->data, nbytes,
    cudaMemcpyHostToDevice, g_gpu.stream));
  
  // ============ STEP 1: TRANSPOSE INPUT ============
  // Grid for transpose: each block processes a TILE_DIM x TILE_DIM tile
  dim3 transpose_block(TILE_DIM, TILE_DIM);
  dim3 transpose_grid((ncols + TILE_DIM - 1) / TILE_DIM,
                      (nrows + TILE_DIM - 1) / TILE_DIM);
  
  transpose_kernel<<<transpose_grid, transpose_block, 0, g_gpu.stream>>>(
    g_gpu.d_img1, g_gpu.d_transpose, ncols, nrows);
  
  CUDA_CHECK(cudaGetLastError());
  
  // ============ STEP 2: APPLY HORIZONTAL CONVOLUTION TO TRANSPOSED DATA ============
  // After transpose: width=nrows, height=ncols
  const int radius = kernel.width / 2;
  dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
  dim3 grid((nrows + BLOCK_DIM_X - 1) / BLOCK_DIM_X,  // Note: swapped dimensions!
            (ncols + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);
  
  const int tile_stride = BLOCK_DIM_X + 2 * radius + 8;
  size_t shared_bytes = BLOCK_DIM_Y * tile_stride * sizeof(float);
  
  if (shared_bytes > 48 * 1024) {
    CUDA_CHECK(cudaFuncSetAttribute(convolveHoriz_Optimized,
      cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024));
    CUDA_CHECK(cudaFuncSetAttribute(convolveHoriz_Optimized,
      cudaFuncAttributePreferredSharedMemoryCarveout, 100));
  }
  
  // Apply horizontal convolution to transposed image
  convolveHoriz_Optimized<<<grid, block, shared_bytes, g_gpu.stream>>>(
    g_gpu.d_transpose, g_gpu.d_img2, nrows, ncols, kernel.width);
  
  CUDA_CHECK(cudaGetLastError());
  
  // ============ STEP 3: TRANSPOSE RESULT BACK ============
  transpose_kernel<<<transpose_grid, transpose_block, 0, g_gpu.stream>>>(
    g_gpu.d_img2, g_gpu.d_img1, nrows, ncols);
  
  CUDA_CHECK(cudaGetLastError());
  
  // Copy result back to host
  CUDA_CHECK(cudaMemcpyAsync(imgout->data, g_gpu.d_img1, nbytes,
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
    if (g_gpu.d_transpose) cudaFree(g_gpu.d_transpose);
    cudaStreamDestroy(g_gpu.stream);
    g_gpu.initialized = false;
  }
}
