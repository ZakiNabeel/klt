/**********************************************************************
Finds the 150 best features in an image and tracks them through the 
next two images.  The sequential mode is set in order to speed
processing.  The features are stored in a feature table, which is then
saved to a text file; each feature list is also written to a PPM file.
**********************************************************************/
#include <stdlib.h>
#include <stdio.h>
#include "pnmio.h"
#include "klt.h"

/* Define data directory - can be overridden at compile time */
#ifndef DATA_DIR
#define DATA_DIR "data/"
#endif

/* Define output directory - can be overridden at compile time */
#ifndef OUTPUT_DIR
#define OUTPUT_DIR "output/"
#endif

/* #define REPLACE */

#ifdef WIN32
int RunExample3()
#else
int main()
#endif
{
  unsigned char *img1, *img2;
  char fnamein[100], fnameout[100];
  KLT_TrackingContext tc;
  KLT_FeatureList fl;
  KLT_FeatureTable ft;
  int nFeatures = 150, nFrames = 448;
  int ncols, nrows;
  int i;
  
  
  char cmd[256];
  sprintf(cmd, "ls %simg*.pgm 2>/dev/null | wc -l", DATA_DIR);
  FILE *fp = popen(cmd, "r");
  fscanf(fp, "%d", &nFrames);
  pclose(fp);

  tc = KLTCreateTrackingContext();
  fl = KLTCreateFeatureList(nFeatures);
  ft = KLTCreateFeatureTable(nFrames, nFeatures);
  tc->sequentialMode = TRUE;
  tc->writeInternalImages = FALSE;
  tc->affineConsistencyCheck = -1;  /* set this to 2 to turn on affine consistency check */
 
  sprintf(fnamein, "%simg0.pgm", DATA_DIR);
  img1 = pgmReadFile(fnamein, NULL, &ncols, &nrows);
  img2 = (unsigned char *) malloc(ncols*nrows*sizeof(unsigned char));
  
  KLTSelectGoodFeatures(tc, img1, ncols, nrows, fl);
  KLTStoreFeatureList(fl, ft, 0);
  sprintf(fnameout, "%sfeat0.ppm", OUTPUT_DIR);
  KLTWriteFeatureListToPPM(fl, img1, ncols, nrows, fnameout);
  
  for (i = 1 ; i < nFrames ; i++)  {
    sprintf(fnamein, "%simg%d.pgm", DATA_DIR, i);
    pgmReadFile(fnamein, img2, &ncols, &nrows);
    KLTTrackFeatures(tc, img1, img2, ncols, nrows, fl);
#ifdef REPLACE
    KLTReplaceLostFeatures(tc, img2, ncols, nrows, fl);
#endif
    KLTStoreFeatureList(fl, ft, i);
    sprintf(fnameout, "%sfeat%d.ppm", OUTPUT_DIR, i);
    KLTWriteFeatureListToPPM(fl, img2, ncols, nrows, fnameout);
  }
  
  sprintf(fnameout, "%sfeatures.txt", OUTPUT_DIR);
  KLTWriteFeatureTable(ft, fnameout, "%5.1f");
  sprintf(fnameout, "%sfeatures.ft", OUTPUT_DIR);
  KLTWriteFeatureTable(ft, fnameout, NULL);
  
  KLTFreeFeatureTable(ft);
  KLTFreeFeatureList(fl);
  KLTFreeTrackingContext(tc);
  free(img1);
  free(img2);
  
  return 0;
}
