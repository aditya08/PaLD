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
    int n, l1_block_size, l2_block_size, conflict_block_size, cohesion_block_size, ntrials, i;

    if ((argc < 7) || !(n = atoi(argv[1])) || !(conflict_block_size = atoi(argv[2])) || !(cohesion_block_size = atoi(argv[3])) || !(l1_block_size = atoi(argv[4])) || !(l2_block_size = atoi(argv[5])) || !(ntrials = atoi(argv[6]))) {
        fprintf(stderr, "Usage: ./name distance_mat_size conflict_block_size cohesion_block_size l1_block_size l2_block_size ntrials\n");
        exit(-1);
    }

    // block_size = argc > 2 ? 2 : atoi(argv[2]);
    // block_size = argc == 2 ? 2 : atoi(argv[2]);

    unsigned int num_gen = n * n;
    float *C1 = _mm_malloc(num_gen*sizeof(float), VECALIGN);
    memset(C1, 0, sizeof(float)*num_gen);
    float *D = _mm_malloc(sizeof(float) * num_gen, VECALIGN);
    dist_mat_gen2D(D, n, 1, 10*n, 12345, '2');

    //print out dist matrix
    // printf("[\n");
    // for (i = 0; i < num_gen; i++) {

    //     if (i % n == 0 && i > 0) {
    //         printf(";\n");
    //     }
    //     printf("%.2f ", D[i]);
    // }
    // printf("]\n");
    // FILE *f = fopen("dist_mat.bin", "wb");
    // fwrite(D, sizeof(float), num_gen, f);
    // fclose(f);
    //computing C with optimal block algorithm
    double start = 0., naive_time = 0.;
    for (int i = 0; i < ntrials; ++i){
        memset(C1, 0, sizeof(float)*n*n);
        start = omp_get_wtime();
        pald_triplet_intrin(D, 1, n, C1, conflict_block_size, cohesion_block_size);
        naive_time += omp_get_wtime() - start;
    }

    float *C2 = _mm_malloc(num_gen*sizeof(float), VECALIGN);
    memset(C2, 0, sizeof(float)*num_gen);
    // print_out(n,C1);
    double opt_time = 0.;
    // printf("%d\n",num_gen);
    for (int i = 0; i < ntrials; ++i){
        memset(C2, 0, sizeof(float)*n*n);
        start = omp_get_wtime();
        pald_triplet_L2_blocked(D, 1, n, C2, l1_block_size, l2_block_size);
        opt_time += omp_get_wtime() - start;
    }
    float d, maxdiff = 0.;
    for (i = 0; i < num_gen; i++) {
        d = fabs(C1[i]-C2[i]);
        maxdiff = d > maxdiff ? d : maxdiff;
    }

    _mm_free(D);
    _mm_free(C2);
    _mm_free(C1);
    // print_out(n,C1);
    //print out block algorithm result
    // print_out(n, C2);
    float *C = _mm_malloc(num_gen*sizeof(float), VECALIGN);
    memset(C, 0, sizeof(float)*num_gen);
    float *A = _mm_malloc(sizeof(float) * num_gen, VECALIGN);
    float *B = _mm_malloc(sizeof(float) * num_gen, VECALIGN);
    sgemm_rand(A, num_gen, -1.f, 1.f, 42);
    sgemm_rand(B, num_gen, -1.f, 1.f, 42);

    double sgemm_time = 0.;
    for(int i = 0; i < ntrials; ++i){
        memset(C, 0, sizeof(float)*num_gen);
        start = omp_get_wtime();
        // cblas_ssyrk(CblasColMajor, CblasUpper, CblasTrans, n, n, 1.f, A, n, 0.f, C, n);
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

    // compute max norm error between two cohesion matrices

    printf("=============================================\n");
    printf("           Summary, n: %d\n", n);
    printf("=============================================\n");
    printf("Triplet Two-Level Blocked time: %.5fs\n",opt_time/ntrials);
    printf("Triplet time: %.5fs\n",naive_time/ntrials);

    printf("Speedup: %.2f\n", naive_time/opt_time);
    printf("Maximum difference: %1.8e\n\n", maxdiff);

    printf("SGEMM time: %.5fs\n", sgemm_time/ntrials);

    //free sgemm matrices.
    _mm_free(A);
    _mm_free(B);
    _mm_free(C);
}