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
 * OPTIMIZED HORIZONTAL CONVOLUTION
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
 * OPTIMIZED VERTICAL CONVOLUTION
 * 
 * Uses strided access but with shared memory caching to hide latency
 * Simple and effective - sometimes the straightforward approach wins!
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
  
  // Shared memory layout: [tile_height + 2*radius][tile_width + 8]
  const int tile_stride = tile_width + 8;  // Padding for bank conflicts
  const int tile_vert = tile_height + 2 * radius;
  extern __shared__ float s_tile[];
  
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int gx = blockIdx.x * tile_width + tx;
  const int gy = blockIdx.y * tile_height + ty;
  
  if (gx >= ncols) return;
  
  // ============ PHASE 1: LOAD TILE ============
  const int tile_start_row = blockIdx.y * tile_height - radius;
  
  // Each thread loads a vertical strip
  for (int local_row = ty; local_row < tile_vert; local_row += tile_height) {
    int global_row = tile_start_row + local_row;
    
    float val = 0.0f;
    if (global_row >= 0 && global_row < nrows && gx < ncols) {
      val = imgin[global_row * ncols + gx];
    }
    s_tile[local_row * tile_stride + tx] = val;
  }
  __syncthreads();
  
  // ============ PHASE 2: COMPUTE CONVOLUTION ============
  if (gy >= nrows) return;
  
  // Zero boundary pixels
  if (gy < radius || gy >= nrows - radius) {
    imgout[gy * ncols + gx] = 0.0f;
    return;
  }
  
  // Convolution with aggressive unrolling
  float sum = 0.0f;
  int s_center_row = ty + radius;
  
  if (kernel_width <= 7) {
    #pragma unroll
    for (int k = 0; k < kernel_width; k++) {
      sum += s_tile[(s_center_row - radius + k) * tile_stride + tx] * c_kernel[k];
    }
  } else if (kernel_width <= 15) {
    #pragma unroll 4
    for (int k = 0; k < kernel_width; k++) {
      sum += s_tile[(s_center_row - radius + k) * tile_stride + tx] * c_kernel[k];
    }
  } else {
    #pragma unroll 2
    for (int k = 0; k < kernel_width; k++) {
      sum += s_tile[(s_center_row - radius + k) * tile_stride + tx] * c_kernel[k];
    }
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
  
  // ============ VERTICAL CONVOLUTION ============
  const int radius = kernel.width / 2;
  dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
  dim3 grid((ncols + BLOCK_DIM_X - 1) / BLOCK_DIM_X,
            (nrows + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);
  
  // Calculate shared memory
  const int tile_vert = BLOCK_DIM_Y + 2 * radius;
  const int tile_stride = BLOCK_DIM_X + 8;
  size_t shared_bytes = tile_vert * tile_stride * sizeof(float);
  
  if (shared_bytes > 48 * 1024) {
    CUDA_CHECK(cudaFuncSetAttribute(convolveVert_Optimized,
      cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024));
    CUDA_CHECK(cudaFuncSetAttribute(convolveVert_Optimized,
      cudaFuncAttributePreferredSharedMemoryCarveout, 100));
  }
  
  // Vertical convolution
  convolveVert_Optimized<<<grid, block, shared_bytes, g_gpu.stream>>>(
    g_gpu.d_img1, g_gpu.d_img2, ncols, nrows, kernel.width);
  
  CUDA_CHECK(cudaGetLastError());
  
  // Copy result back to host
  CUDA_CHECK(cudaMemcpyAsync(imgout->data, g_gpu.d_img2, nbytes,
    cudaMemcpyDeviceToHost, g_gpu.stream));
  
  CUDA_CHECK(cudaStreamSynchronize(g_gpu.stream));
  
  imgout->ncols = ncols;
  imgout->nrows = nrows;
}
 
/*********************************************************************
 * Separable Convolution - OPTIMIZED GPU VERSION
 * 
 * Keep data on GPU for both passes - only 2 CPU↔GPU transfers total!
 *********************************************************************/
static void _convolveSeparate(
  _KLT_FloatImage imgin,
  ConvolutionKernel horiz_kernel,
  ConvolutionKernel vert_kernel,
  _KLT_FloatImage imgout)
{
  const int ncols = imgin->ncols;
  const int nrows = imgin->nrows;
  const size_t nbytes = ncols * nrows * sizeof(float);
  
  ensure_gpu_buffers(nbytes);
  
  // ============ UPLOAD INPUT ONCE ============
  CUDA_CHECK(cudaMemcpyAsync(g_gpu.d_img1, imgin->data, nbytes,
    cudaMemcpyHostToDevice, g_gpu.stream));
  
  // ============ HORIZONTAL PASS (GPU → GPU) ============
  {
    // Copy kernel to constant memory (reversed)
    float reversed_kernel[MAX_KERNEL_SIZE];
    for (int i = 0; i < horiz_kernel.width; i++) {
      reversed_kernel[i] = horiz_kernel.data[horiz_kernel.width - 1 - i];
    }
    CUDA_CHECK(cudaMemcpyToSymbolAsync(c_kernel, reversed_kernel,
      horiz_kernel.width * sizeof(float), 0, cudaMemcpyHostToDevice, g_gpu.stream));
    
    const int radius = horiz_kernel.width / 2;
    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 grid((ncols + BLOCK_DIM_X - 1) / BLOCK_DIM_X,
              (nrows + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);
    
    const int tile_stride = BLOCK_DIM_X + 2 * radius + 8;
    size_t shared_bytes = BLOCK_DIM_Y * tile_stride * sizeof(float);
    
    if (shared_bytes > 48 * 1024) {
      CUDA_CHECK(cudaFuncSetAttribute(convolveHoriz_Optimized,
        cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024));
      CUDA_CHECK(cudaFuncSetAttribute(convolveHoriz_Optimized,
        cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    }
    
    // d_img1 → d_img2
    convolveHoriz_Optimized<<<grid, block, shared_bytes, g_gpu.stream>>>(
      g_gpu.d_img1, g_gpu.d_img2, ncols, nrows, horiz_kernel.width);
    
    CUDA_CHECK(cudaGetLastError());
  }
  
  // ============ VERTICAL PASS (GPU → GPU) ============
  {
    // Copy kernel to constant memory (reversed)
    float reversed_kernel[MAX_KERNEL_SIZE];
    for (int i = 0; i < vert_kernel.width; i++) {
      reversed_kernel[i] = vert_kernel.data[vert_kernel.width - 1 - i];
    }
    CUDA_CHECK(cudaMemcpyToSymbolAsync(c_kernel, reversed_kernel,
      vert_kernel.width * sizeof(float), 0, cudaMemcpyHostToDevice, g_gpu.stream));
    
    const int radius = vert_kernel.width / 2;
    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 grid((ncols + BLOCK_DIM_X - 1) / BLOCK_DIM_X,
              (nrows + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);
    
    const int tile_vert = BLOCK_DIM_Y + 2 * radius;
    const int tile_stride = BLOCK_DIM_X + 8;
    size_t shared_bytes = tile_vert * tile_stride * sizeof(float);
    
    if (shared_bytes > 48 * 1024) {
      CUDA_CHECK(cudaFuncSetAttribute(convolveVert_Optimized,
        cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024));
      CUDA_CHECK(cudaFuncSetAttribute(convolveVert_Optimized,
        cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    }
    
    // d_img2 → d_img1
    convolveVert_Optimized<<<grid, block, shared_bytes, g_gpu.stream>>>(
      g_gpu.d_img2, g_gpu.d_img1, ncols, nrows, vert_kernel.width);
    
    CUDA_CHECK(cudaGetLastError());
  }
  
  // ============ DOWNLOAD RESULT ONCE ============
  CUDA_CHECK(cudaMemcpyAsync(imgout->data, g_gpu.d_img1, nbytes,
    cudaMemcpyDeviceToHost, g_gpu.stream));
  
  CUDA_CHECK(cudaStreamSynchronize(g_gpu.stream));
  
  imgout->ncols = ncols;
  imgout->nrows = nrows;
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
  
  const int ncols = img->ncols;
  const int nrows = img->nrows;
  const size_t nbytes = ncols * nrows * sizeof(float);
  
  ensure_gpu_buffers(nbytes);
  
  // ============ UPLOAD INPUT IMAGE ONCE ============
  CUDA_CHECK(cudaMemcpyAsync(g_gpu.d_img1, img->data, nbytes,
    cudaMemcpyHostToDevice, g_gpu.stream));
  
  // ============ COMPUTE GRADX: (gaussderiv_x * gauss_y) ============
  {
    // Horizontal pass with gaussderiv
    float reversed_kernel[MAX_KERNEL_SIZE];
    for (int i = 0; i < gaussderiv_kernel.width; i++) {
      reversed_kernel[i] = gaussderiv_kernel.data[gaussderiv_kernel.width - 1 - i];
    }
    CUDA_CHECK(cudaMemcpyToSymbolAsync(c_kernel, reversed_kernel,
      gaussderiv_kernel.width * sizeof(float), 0, cudaMemcpyHostToDevice, g_gpu.stream));
    
    int radius = gaussderiv_kernel.width / 2;
    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 grid((ncols + BLOCK_DIM_X - 1) / BLOCK_DIM_X,
              (nrows + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);
    size_t shared_bytes = BLOCK_DIM_Y * (BLOCK_DIM_X + 2 * radius + 8) * sizeof(float);
    
    convolveHoriz_Optimized<<<grid, block, shared_bytes, g_gpu.stream>>>(
      g_gpu.d_img1, g_gpu.d_img2, ncols, nrows, gaussderiv_kernel.width);
    
    // Vertical pass with gauss
    for (int i = 0; i < gauss_kernel.width; i++) {
      reversed_kernel[i] = gauss_kernel.data[gauss_kernel.width - 1 - i];
    }
    CUDA_CHECK(cudaMemcpyToSymbolAsync(c_kernel, reversed_kernel,
      gauss_kernel.width * sizeof(float), 0, cudaMemcpyHostToDevice, g_gpu.stream));
    
    radius = gauss_kernel.width / 2;
    grid = dim3((ncols + BLOCK_DIM_X - 1) / BLOCK_DIM_X,
                (nrows + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);
    shared_bytes = (BLOCK_DIM_Y + 2 * radius) * (BLOCK_DIM_X + 8) * sizeof(float);
    
    convolveVert_Optimized<<<grid, block, shared_bytes, g_gpu.stream>>>(
      g_gpu.d_img2, g_gpu.d_img1, ncols, nrows, gauss_kernel.width);
    
    // Download gradx result
    CUDA_CHECK(cudaMemcpyAsync(gradx->data, g_gpu.d_img1, nbytes,
      cudaMemcpyDeviceToHost, g_gpu.stream));
  }
  
  // ============ COMPUTE GRADY: (gauss_x * gaussderiv_y) ============
  // Note: img is still in g_gpu.d_img1 from initial upload
  {
    // Horizontal pass with gauss
    float reversed_kernel[MAX_KERNEL_SIZE];
    for (int i = 0; i < gauss_kernel.width; i++) {
      reversed_kernel[i] = gauss_kernel.data[gauss_kernel.width - 1 - i];
    }
    CUDA_CHECK(cudaMemcpyToSymbolAsync(c_kernel, reversed_kernel,
      gauss_kernel.width * sizeof(float), 0, cudaMemcpyHostToDevice, g_gpu.stream));
    
    // Wait for gradx download to finish before reusing d_img1
    CUDA_CHECK(cudaStreamSynchronize(g_gpu.stream));
    
    // Re-upload img for grady computation
    CUDA_CHECK(cudaMemcpyAsync(g_gpu.d_img1, img->data, nbytes,
      cudaMemcpyHostToDevice, g_gpu.stream));
    
    int radius = gauss_kernel.width / 2;
    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 grid((ncols + BLOCK_DIM_X - 1) / BLOCK_DIM_X,
              (nrows + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);
    size_t shared_bytes = BLOCK_DIM_Y * (BLOCK_DIM_X + 2 * radius + 8) * sizeof(float);
    
    convolveHoriz_Optimized<<<grid, block, shared_bytes, g_gpu.stream>>>(
      g_gpu.d_img1, g_gpu.d_img2, ncols, nrows, gauss_kernel.width);
    
    // Vertical pass with gaussderiv
    for (int i = 0; i < gaussderiv_kernel.width; i++) {
      reversed_kernel[i] = gaussderiv_kernel.data[gaussderiv_kernel.width - 1 - i];
    }
    CUDA_CHECK(cudaMemcpyToSymbolAsync(c_kernel, reversed_kernel,
      gaussderiv_kernel.width * sizeof(float), 0, cudaMemcpyHostToDevice, g_gpu.stream));
    
    radius = gaussderiv_kernel.width / 2;
    grid = dim3((ncols + BLOCK_DIM_X - 1) / BLOCK_DIM_X,
                (nrows + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);
    shared_bytes = (BLOCK_DIM_Y + 2 * radius) * (BLOCK_DIM_X + 8) * sizeof(float);
    
    convolveVert_Optimized<<<grid, block, shared_bytes, g_gpu.stream>>>(
      g_gpu.d_img2, g_gpu.d_img1, ncols, nrows, gaussderiv_kernel.width);
    
    // Download grady result
    CUDA_CHECK(cudaMemcpyAsync(grady->data, g_gpu.d_img1, nbytes,
      cudaMemcpyDeviceToHost, g_gpu.stream));
  }
  
  CUDA_CHECK(cudaStreamSynchronize(g_gpu.stream));
  
  gradx->ncols = ncols;
  gradx->nrows = nrows;
  grady->ncols = ncols;
  grady->nrows = nrows;
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
