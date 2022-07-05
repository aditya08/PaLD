#ifndef PALD_KERNELS_H
#define PALD_KERNELS_H

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <omp.h>

// pald algorithms
void pald_orig(float *D, float beta, int n, float *C);
void pald_opt(float *D, float beta, int n, float *C, const int b);
void pald_opt_par(float *D, float beta, int n, float *C, const int b, int num_threads);
void pald_opt_new(float *D, float beta, int n, float *C);
void pald(float *D, float beta, int n, float *C, int block_size);
void pald_openmp(float *D, float beta, int n, float *C, int block_size, int num_threads);

#endif //PALD_KERNELS_H
