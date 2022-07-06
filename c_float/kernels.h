#ifndef PALD_KERNELS_H
#define PALD_KERNELS_H

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <omp.h>

// pald algorithms that loop over all z.
void pald_allz_orig(float *D, float beta, int n, float *C);
void pald_allz_orig_openmp(float *D, float beta, int n, float *C, const int b, int num_threads);
void pald_allz(float * restrict D, float beta, int n, float * restrict C, int block_size);
void pald_allz_openmp(float * restrict D, float beta, int n, float * restrict C, int block_size, int num_threads);

// PaLD algorithms that loop over triplets.
void pald_triplet_orig(float *D, float beta, int n, float *C);
void pald_triplet(float *D, float beta, int n, float *C);
void pald_triplet_openmp(float *D, float beta, int n, float *C);
#endif //PALD_KERNELS_H
