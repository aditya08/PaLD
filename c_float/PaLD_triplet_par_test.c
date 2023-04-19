//
// Created by Aditya Devarakonda on 7/12/2022.
//
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "kernels.h"
#include "utils.h"

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
    printf("];\n");
}

void print_diag(int n, float *C){
    printf("\n");
    int i , j;
    register int temp;
    register float c;
    printf("[\n");
    for (i = 0; i < n; i++) {
        temp = i * n + i;
        c = C[temp]/(n - 1);
        printf("%.8f ", c);
    }
    printf("]\n");
}

int main(int argc, char **argv) {

    //initializing testing environment spec
    int n, seq_block_size, i, nthreads, omp_block_size, ntrials;

    if ((argc != 6) || !(n = atoi(argv[1])) || !(seq_block_size = atoi(argv[2])) || !(omp_block_size = atoi(argv[3])) ||  !(nthreads = atoi(argv[4])) || !(ntrials = atoi(argv[5]))) {
        fprintf(stderr, "Usage: ./name distance_mat_size sequential_block_size openmp_block_size num_threads num_trials\n");
        exit(-1);
    }

    unsigned int num_gen = n * n;

    float *C1 = _mm_malloc(num_gen*sizeof(float), VECALIGN);
    float *C2 = _mm_malloc(num_gen*sizeof(float), VECALIGN);
    memset(C1, 0, sizeof(float)*num_gen);
    memset(C2, 0, sizeof(float)*num_gen);
    float *D = _mm_malloc(sizeof(float) * num_gen, VECALIGN);
    dist_mat_gen2D(D, n, 1, 10*n, 12345, '2');

    //print out dist matrix
    // printf("[\n");
    // for (i = 0; i < num_gen; i++) {

    //     if (i % n == 0 && i > 0) {
    //         printf(";\n");
    //     }
    //     printf("%.7f ", D[i]);
    // }
    // printf("];\n");
    // FILE *f = fopen("dist_mat.bin", "wb");
    // fwrite(D, sizeof(float), num_gen, f);
    // fclose(f);
    //computing C with optimal block algorithm
    for(int i = 0; i < 3; ++i){
        memset(C1, 0, sizeof(float)*n*n);
        pald_triplet_intrin(D, 1., n, C1, seq_block_size);
    }
    double start = 0., naive_time = 0., omp_time = 0.;
    for (int i = 0; i < ntrials; ++i){
        memset(C1, 0, sizeof(float)*n*n);
        start = omp_get_wtime();
        //pald_triplet_naive(D, 1, n, C1);
        pald_triplet_intrin(D, 1., n, C1, seq_block_size);
        // pald_triplet_L2_blocked(D, 1., n, C1, seq_block_size, 512);
        // pald_allz(D, 1, n, C1, 256);
        naive_time += omp_get_wtime() - start;
    }

    for(int i = 0; i < 3; ++i){
        memset(C2, 0, sizeof(float)*n*n);
        pald_triplet_openmp(D, 1., n, C2, omp_block_size, nthreads);
    }
    //print_out(n, C1);
    for (int i = 0; i < ntrials; ++i){
        memset(C2, 0, sizeof(float)*n*n);
        start = omp_get_wtime();
        // pald_triplet_blocked(D, 1, n, C2, nthreads);
        //pald_triplet_naive(D, 1, n, C2);
        pald_triplet_openmp(D, 1., n, C2, omp_block_size, nthreads);
        //pald_triplet_naive_openmp(D, 1, n, C2, nthreads);
        omp_time += omp_get_wtime() - start;
    }


    //print out block algorithm result
    //print_out(n, C1);
    // print_diag(n, C1);
    // printf("\n\n");
    //print_diag(n, C2);


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

    float d, maxdiff = 0.;
    for (i = 0; i < num_gen; i++) {
        d = fabs(C1[i]-C2[i]);
        //printf("(%.8f, %.8f), diff: %.8e\n", C1[i*n + i], C2[i * n + i], d);
        maxdiff = d > maxdiff ? d : maxdiff;
    }
    // printf("Maximum difference: %1.1e \n", maxdiff);

    printf("=============================================\n");
    printf("           Summary, n: %d\n", n);
    printf("=============================================\n");
    printf("Avg. Sequential time: %.5fs\n",naive_time/ntrials);
    printf("Avg. Parallel   time: %.5fs, nthreads: %d\n", omp_time/ntrials, nthreads);
    printf("Speedup: %.2f\n",naive_time/omp_time);
    printf("Parallel Efficiency: %2.2f\n", naive_time/omp_time/nthreads*100);
    printf("Maximum difference: %1.8e\n\n", maxdiff);
    printf("triplet ops: %e Gops\n\n", triplet_ops(n, seq_block_size)*10e-9);
    printf("triplet avg. ops/sec: %e Gops/sec\n\n", triplet_ops(n, seq_block_size)*10e-9/(naive_time/ntrials));
    printf("gemm ops: %e Gflops\n\n", 10e-9*n*n*n);

    _mm_free(D);
    _mm_free(C2);
    _mm_free(C1);
}