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
    unsigned int n, block_size, i, ntrials;

    if ((argc != 4) || !(n = atoi(argv[1])) || !(block_size = atoi(argv[2])) || !(ntrials = atoi(argv[3]))) {
        fprintf(stderr, "Usage: ./name distance_mat_size block_size num_trials\n");
        exit(-1);
    }

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
    //     printf("%.7f ", D[i]);
    // }
    // printf("];\n");
    // FILE *f = fopen("dist_mat.bin", "wb");
    // fwrite(D, sizeof(float), num_gen, f);
    // fclose(f);
    //computing C with optimal block algorithm
    // for (unsigned int i = 0; i < 3; ++i){
    //     memset(C1, 0, sizeof(float)*n*n);
    //     // start = omp_get_wtime();
    //     pald_triplet_intrin(D, 1., n, C1, seq_block_size);
    //     // pald_triplet_largezblock(D, 1., n, C1, seq_block_size, 1024);
    //     // naive_time += omp_get_wtime() - start;
    // }
    for (unsigned int i = 0; i < 0; ++i){
        memset(C1, 0, sizeof(float)*n*n);
        pald_allz_noties_nobeta_vecbranching(D, 1.0, n, C1, block_size);
        // pald_triplet_largezblock(D, 1., n, C1, seq_block_size, 1024);
    }


    double start = 0., naive_time = 0.;
    for (unsigned int i = 0; i < ntrials; ++i){
        memset(C1, 0, sizeof(float)*n*n);
        start = omp_get_wtime();
        pald_allz_noties_nobeta_vecbranching(D, 1.0, n, C1, block_size);
        // pald_triplet_largezblock(D, 1., n, C1, seq_block_size, 1024);
        naive_time += omp_get_wtime() - start;
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

    printf("=============================================\n");
    printf("           Summary, n: %d\n", n);
    printf("=============================================\n");
    printf("Avg. Sequential time: %.5fs\n",naive_time/ntrials);

    // printf("gemm avg. Gflops/sec: %e\n", (10e-9*n*n*n));

    _mm_free(D);
    _mm_free(C1);
}