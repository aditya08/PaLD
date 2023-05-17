//
// Created by Aditya Devarakonda on 7/12/2022.
//
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "kernels.h"
#include "utils.h"
#include "omp.h"
#include "mkl.h"

void print_out(int n, float *C) {
    printf("\n");
    int i, j;
    register int temp;
    printf("[\n");
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            temp = j * n + i;
            C[temp] /= (n - 1);
            printf("%.7f ", C[temp]);
        }
        printf(";\n");
    }
    printf("]\n");
}

int main(int argc, char **argv) {

    //initializing testing environment spec
    int n, triplet_L2_cache_size, conflict_block_size, cohesion_block_size, allz_cache_size, i;
    int ntrials;
    if ((argc < 7) || !(n = atoi(argv[1])) || !(conflict_block_size = atoi(argv[2])) || !(cohesion_block_size = atoi(argv[3])) || !(triplet_L2_cache_size = atoi(argv[4])) || !(allz_cache_size = atoi(argv[5])) || !(ntrials = atoi(argv[6]))) {
        fprintf(stderr, "Usage: ./name distance_mat_size conflict_block_size cohesion_block_size triplet_L2_block_size allz_block_size ntrials\n");
        exit(-1);
    }

    // cache_size = argc > 2 ? 2 : atoi(argv[2]);
    // cache_size = argc == 2 ? 2 : atoi(argv[2]);

    unsigned int num_gen = n * n;

    float *C1 = _mm_malloc(num_gen*sizeof(float), VECALIGN);
    float *C2 = _mm_malloc(num_gen*sizeof(float), VECALIGN);
    memset(C1, 0, sizeof(float)*num_gen);
    memset(C2, 0, sizeof(float)*num_gen);
    float *D = _mm_malloc(sizeof(float) * num_gen, VECALIGN);
    dist_mat_gen2D(D, n, 1, 10*n, 12345, '2');

    // FILE *f = fopen("dist_mat.bin", "wb");
    // fwrite(D, sizeof(float), num_gen, f);
    // fclose(f);
    //computing C with optimal block algorithm
    double start = 0., naive_time = 0.;
    for (int i = 0; i < ntrials; ++i){
        memset(C1, 0, sizeof(float)*n*n);
        start = omp_get_wtime();
        // pald_triplet_L2_blocked(D, 1, n, C1, triplet_L1_cache_size,triplet_L2_cache_size);
        // pald_triplet(D, 1., n, C1, triplet_L1_cache_size);
        pald_allz(D, 1.0, n, C1, allz_cache_size);
        // pald_allz_openmp(D, 1.0, n, C1, allz_cache_size, 8);
        // pald_triplet_intrin(D, 1, n, C1, allz_cache_size);
        // pald_allz_naive(D, 1.0, n, C1);
        // pald_triplet_naive(D, 1., n, C1);
        naive_time += omp_get_wtime() - start;
    }
    // print_out(n,C1);
    double opt_time = 0.;
    for (int i = 0; i < ntrials; ++i){
        memset(C2, 0, sizeof(float)*n*n);
        start = omp_get_wtime();
        // pald_allz_openmp_noties_nobeta_vecbranching(D, 1.0, n, C2, allz_cache_size, 16);
        // pald_triplet_fewercompares(D, 1, n, C2, triplet_L1_cache_size);
        // pald_triplet(D, 1, n, C2, triplet_L1_cache_size);
        pald_triplet_intrin(D, 1, n, C2, conflict_block_size);
        // pald_triplet_L2_blocked(D, 1, n, C2, triplet_L1_cache_size, triplet_L2_cache_size);
        // pald_triplet_intrin_powersoftwo(D, 1., n, C2, triplet_L1_cache_size);
        // pald_triplet_largezblock(D, 1., n, C2, triplet_L1_cache_size, triplet_L2_cache_size);
        opt_time += omp_get_wtime() - start;
    }
    //print out triplet algorithms result
    // print_out(n,C2);
    // print_out(n, C1);

    // compute max norm error between two cohesion matrices
    float d, maxdiff = 0.;
    for (i = 0; i < num_gen; i++) {
        d = fabs(C1[i]-C2[i]);
        maxdiff = d > maxdiff ? d : maxdiff;
    }

    _mm_free(D);
    _mm_free(C2);
    _mm_free(C1);


    float *C = _mm_malloc(num_gen*sizeof(float), VECALIGN);
    memset(C, 0, sizeof(float)*num_gen);
    float *A = _mm_malloc(sizeof(float) * num_gen, VECALIGN);
    float *B = _mm_malloc(sizeof(float) * num_gen, VECALIGN);
    sgemm_rand(A, num_gen, -1.f, 1.f, 42);
    sgemm_rand(B, num_gen, -1.f, 1.f, 42);


    double sgemm_time = 0.;
    for(int i = 0; i < ntrials; ++i){
        memset(C, 0, sizeof(float)*num_gen);
        omp_set_num_threads(1);
        start = omp_get_wtime();
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.f, A, n, B, n, 0.f, C, n);
        sgemm_time += omp_get_wtime() - start;
    }

    //computing C with original algorithm

    // start = clock();
    // //for (int i = 0; i < 4; ++i)
    // pald_allz_naive(D, 1, n, C2);
    // diff = clock() - start;
    // double msec_orig = 1. * diff / CLOCKS_PER_SEC;


    //print out result of original algorithm
    //print_out(n, C);
    // print out for error checking


    printf("=============================================\n");
    printf("           Summary, n: %d\n", n);
    printf("=============================================\n");
    printf("Triplet skip-ties + int-ops + baseline time: %.5fs\n",naive_time/ntrials);
    printf("Triplet L2-blocked, skip-ties, int-ops time: %.5fs\n",opt_time/ntrials);
    // printf("Allz experimental time: %.5fs\n",naive_time/ntrials);
    // printf("Allz optimized time: %.5fs\n",opt_time/ntrials);

    printf("Speedup: %.2f\n", naive_time/opt_time);
    printf("Maximum difference: %1.8e\n\n", maxdiff);

    printf("SGEMM time: %.5fs\n", sgemm_time/ntrials);
    printf("SGEMM Speedup: %.2f\n", naive_time/sgemm_time);
    //free sgemm matrices.
    _mm_free(A);
    _mm_free(B);
    _mm_free(C);
}
