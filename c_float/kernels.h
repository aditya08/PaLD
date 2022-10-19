#ifndef PALD_KERNELS_H
#define PALD_KERNELS_H

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <omp.h>

// pald algorithms that loop over all z.
void pald_allz_naive(float *D, float beta, int n, float *C);
void pald_allz_naive_openmp(float *D, float beta, int n, float *C, const int b, int num_threads);
void pald_allz(float * restrict D, float beta, int n, float * restrict C, int block_size);
void pald_allz_experimental(float * restrict D, float beta, int n, float * restrict C, int block_size);
void pald_allz_openmp(float * restrict D, float beta, int n, float * restrict C, int block_size, int num_threads);

// PaLD algorithms that loop over triplets.
void pald_triplet_naive(float *D, float beta, int n, float *C);
void pald_triplet_naive_openmp(float *D, float beta, int n, float *C, int num_threads);
void pald_triplet_blocked(float *D, float beta, int n, float *C, int block_size);
void pald_triplet(float *D, float beta, int n, float *C, int block_size);
void pald_triplet_intrin(float *D, float beta, int n, float *C, int block_size);
void pald_triplet_remainder_loop(float *D, float beta, int n, float *C, int block_size);
void pald_triplet_openmp(float *D, float beta, int n, float *C, int block_size, int num_threads);
void pald_triplet_L2_blocked(float* restrict D, float beta, int n, float* restrict C, int block_size, int l2_block_size);

void pald_triplet_intrin_openmp(float *D, float beta, int n, float *C, int block_size);

#endif //PALD_KERNELS_H
