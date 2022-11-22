#ifndef PALD_UTILS_H
#define PALD_UTILS_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
 
void sym_mat_gen(float *D, const int edge_len, const int upper_limit, const unsigned int seed);
void point_gen2D(float* x, float* y, int n, int min, int max);
void dist_mat_gen2D(float *D, int n, int min, int max, int seed, char dist);
void sgemm_rand(float *A, int len, float min, float max, int seed);
float triplet_ops(int n, int block_size);

#endif //PALD_UTILS_H
