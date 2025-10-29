/*********************************************************************
 * convolve_cuda.cu - CORRECTLY MATCHING CPU VERSION
 *********************************************************************/

/* Standard includes */
#include <assert.h>
#include <math.h>
#include <stdlib.h>

/* CUDA includes */
#include <cuda_runtime.h>

/* Our includes */
#include "base.h"
#include "error.h"
#include "convolve.h"
#include "klt_util.h"

#define MAX_KERNEL_WIDTH 	71

typedef struct  {
  int width;
  float data[MAX_KERNEL_WIDTH];
}  ConvolutionKernel;

/* Kernels */
static ConvolutionKernel gauss_kernel;
static ConvolutionKernel gaussderiv_kernel;
static float sigma_last = -10.0;

/*********************************************************************
 * CUDA Error Checking Macro
 *********************************************************************/
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
 * CUDA Kernels - EXACTLY MATCHING CPU BEHAVIOR
 *********************************************************************/

__global__ void _convolveImageHoriz_Kernel(
  float *imgin,
  float *kernel,
  int kernel_width,
  int ncols, int nrows,
  float *imgout)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_pixels = ncols * nrows;
  
  if (idx >= total_pixels) return;
  
  int i = idx % ncols;
  int j = idx / ncols;
  int radius = kernel_width / 2;
  
  // CRITICAL: Match CPU version - zero leftmost and rightmost columns
  if (i < radius || i >= ncols - radius) {
    imgout[idx] = 0.0f;
    return;
  }
  
  // Convolve middle columns - MUST match CPU loop order
  // CPU does: for (k = kernel.width-1 ; k >= 0 ; k--)
  //           sum += *ppp++ * kernel.data[k];
  // where ppp starts at (i - radius)
  
  float sum = 0.0f;
  int base_idx = j * ncols + (i - radius);
  for (int k = kernel_width - 1; k >= 0; k--) {
    sum += imgin[base_idx + (kernel_width - 1 - k)] * kernel[k];
  }
  
  imgout[idx] = sum;
}

__global__ void _convolveImageVert_Kernel(
  float *imgin,
  float *kernel,
  int kernel_width,
  int ncols, int nrows,
  float *imgout)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_pixels = ncols * nrows;
  
  if (idx >= total_pixels) return;
  
  int i = idx % ncols;
  int j = idx / ncols;
  int radius = kernel_width / 2;
  
  // CRITICAL: Match CPU version - zero topmost and bottommost rows
  if (j < radius || j >= nrows - radius) {
    imgout[idx] = 0.0f;
    return;
  }
  
  // Convolve middle rows - MUST match CPU loop order
  // CPU does: for (k = kernel.width-1 ; k >= 0 ; k--)
  //           sum += *ppp * kernel.data[k]; ppp += ncols;
  // where ppp starts at (j - radius) * ncols + i
  
  float sum = 0.0f;
  int base_idx = (j - radius) * ncols + i;
  for (int k = kernel_width - 1; k >= 0; k--) {
    sum += imgin[base_idx + (kernel_width - 1 - k) * ncols] * kernel[k];
  }
  
  imgout[idx] = sum;
}

/*********************************************************************
 * _KLTToFloatImage
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

  while (img < ptrend)  *ptrout++ = (float) *img++;
}

/*********************************************************************
 * _computeKernels
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
    float max_gauss = 1.0f, max_gaussderiv = (float) (sigma*exp(-0.5f));
	
    for (i = -hw ; i <= hw ; i++)  {
      gauss->data[i+hw]      = (float) exp(-i*i / (2*sigma*sigma));
      gaussderiv->data[i+hw] = -i * gauss->data[i+hw];
    }

    gauss->width = MAX_KERNEL_WIDTH;
    for (i = -hw ; fabs(gauss->data[i+hw] / max_gauss) < factor ; 
         i++, gauss->width -= 2);
    gaussderiv->width = MAX_KERNEL_WIDTH;
    for (i = -hw ; fabs(gaussderiv->data[i+hw] / max_gaussderiv) < factor ; 
         i++, gaussderiv->width -= 2);
    if (gauss->width == MAX_KERNEL_WIDTH || 
        gaussderiv->width == MAX_KERNEL_WIDTH)
      KLTError("(_computeKernels) MAX_KERNEL_WIDTH %d is too small for "
               "a sigma of %f", MAX_KERNEL_WIDTH, sigma);
  }

  for (i = 0 ; i < gauss->width ; i++)
    gauss->data[i] = gauss->data[i+(MAX_KERNEL_WIDTH-gauss->width)/2];
  for (i = 0 ; i < gaussderiv->width ; i++)
    gaussderiv->data[i] = gaussderiv->data[i+(MAX_KERNEL_WIDTH-gaussderiv->width)/2];

  {
    const int hw = gaussderiv->width / 2;
    float den;
			
    den = 0.0;
    for (i = 0 ; i < gauss->width ; i++)  den += gauss->data[i];
    for (i = 0 ; i < gauss->width ; i++)  gauss->data[i] /= den;
    den = 0.0;
    for (i = -hw ; i <= hw ; i++)  den -= i*gaussderiv->data[i+hw];
    for (i = -hw ; i <= hw ; i++)  gaussderiv->data[i+hw] /= den;
  }

  sigma_last = sigma;
}

/*********************************************************************
 * _KLTGetKernelWidths
 *********************************************************************/

void _KLTGetKernelWidths(
  float sigma,
  int *gauss_width,
  int *gaussderiv_width)
{
  _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);
  *gauss_width = gauss_kernel.width;
  *gaussderiv_width = gaussderiv_kernel.width;
}

/*********************************************************************
 * _convolveImageHoriz
 *********************************************************************/

static void _convolveImageHoriz(
  _KLT_FloatImage imgin,
  ConvolutionKernel kernel,
  _KLT_FloatImage imgout)
{
  int ncols = imgin->ncols, nrows = imgin->nrows;
  int total_pixels = ncols * nrows;
  
  assert(kernel.width % 2 == 1);
  assert(imgin != imgout);
  assert(imgout->ncols >= imgin->ncols);
  assert(imgout->nrows >= imgin->nrows);

  imgout->ncols = ncols;
  imgout->nrows = nrows;

  float *d_imgin, *d_imgout, *d_kernel;
  CUDA_CHECK(cudaMalloc(&d_imgin, total_pixels * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_imgout, total_pixels * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_kernel, kernel.width * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_imgin, imgin->data, total_pixels * sizeof(float), 
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_kernel, kernel.data, kernel.width * sizeof(float), 
                        cudaMemcpyHostToDevice));

  int blockSize = 256;
  int numBlocks = (total_pixels + blockSize - 1) / blockSize;
  _convolveImageHoriz_Kernel<<<numBlocks, blockSize>>>(
    d_imgin, d_kernel, kernel.width, ncols, nrows, d_imgout);
  
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(imgout->data, d_imgout, total_pixels * sizeof(float), 
                        cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_imgin));
  CUDA_CHECK(cudaFree(d_imgout));
  CUDA_CHECK(cudaFree(d_kernel));
}

/*********************************************************************
 * _convolveImageVert
 *********************************************************************/

static void _convolveImageVert(
  _KLT_FloatImage imgin,
  ConvolutionKernel kernel,
  _KLT_FloatImage imgout)
{
  int ncols = imgin->ncols, nrows = imgin->nrows;
  int total_pixels = ncols * nrows;
  
  assert(kernel.width % 2 == 1);
  assert(imgin != imgout);
  assert(imgout->ncols >= imgin->ncols);
  assert(imgout->nrows >= imgin->nrows);

  imgout->ncols = ncols;
  imgout->nrows = nrows;

  float *d_imgin, *d_imgout, *d_kernel;
  CUDA_CHECK(cudaMalloc(&d_imgin, total_pixels * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_imgout, total_pixels * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_kernel, kernel.width * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_imgin, imgin->data, total_pixels * sizeof(float), 
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_kernel, kernel.data, kernel.width * sizeof(float), 
                        cudaMemcpyHostToDevice));

  int blockSize = 256;
  int numBlocks = (total_pixels + blockSize - 1) / blockSize;
  _convolveImageVert_Kernel<<<numBlocks, blockSize>>>(
    d_imgin, d_kernel, kernel.width, ncols, nrows, d_imgout);
  
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(imgout->data, d_imgout, total_pixels * sizeof(float), 
                        cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_imgin));
  CUDA_CHECK(cudaFree(d_imgout));
  CUDA_CHECK(cudaFree(d_kernel));
}

/*********************************************************************
 * _convolveSeparate
 *********************************************************************/

static void _convolveSeparate(
  _KLT_FloatImage imgin,
  ConvolutionKernel horiz_kernel,
  ConvolutionKernel vert_kernel,
  _KLT_FloatImage imgout)
{
  _KLT_FloatImage tmpimg;
  tmpimg = _KLTCreateFloatImage(imgin->ncols, imgin->nrows);
  
  _convolveImageHoriz(imgin, horiz_kernel, tmpimg);
  _convolveImageVert(tmpimg, vert_kernel, imgout);

  _KLTFreeFloatImage(tmpimg);
}

/*********************************************************************
 * _KLTComputeGradients
 *********************************************************************/

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
	
  _convolveSeparate(img, gaussderiv_kernel, gauss_kernel, gradx);
  _convolveSeparate(img, gauss_kernel, gaussderiv_kernel, grady);
}

/*********************************************************************
 * _KLTComputeSmoothedImage
 *********************************************************************/

void _KLTComputeSmoothedImage(
  _KLT_FloatImage img,
  float sigma,
  _KLT_FloatImage smooth)
{
  assert(smooth->ncols >= img->ncols);
  assert(smooth->nrows >= img->nrows);

  if (fabs(sigma - sigma_last) > 0.05)
    _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);

  _convolveSeparate(img, gauss_kernel, gauss_kernel, smooth);
}