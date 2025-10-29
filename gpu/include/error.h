/*********************************************************************
 * error.h
 *********************************************************************/

#ifndef _ERROR_H_
#define _ERROR_H_

#include <stdio.h>
#include <stdarg.h>

#ifdef __cplusplus
extern "C" {
#endif

void KLTError(char *fmt, ...);
void KLTWarning(char *fmt, ...);

#ifdef __cplusplus
}
#endif

#endif

