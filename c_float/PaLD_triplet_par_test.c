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
    printf("]\n");
}

int main(int argc, char **argv) {

    //initializing testing environment spec
    int n, cache_size, i, nthreads;
    
    if ((argc != 2 && argc != 3) || !(n = atoi(argv[1])) || !(nthreads = atoi(argv[2]))) {
        fprintf(stderr, "Usage: ./name distance_mat_size num_threads\n");
        exit(-1);
    }

    cache_size = argc == 2 ? 2 : atoi(argv[2]);

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
    //     printf("%.2f ", D[i]);
    // }
    // printf("]\n");
    FILE *f = fopen("dist_mat.bin", "wb");
    fwrite(D, sizeof(float), num_gen, f);
    fclose(f);
    int ntrials = 5;
    //computing C with optimal block algorithm
    double start = 0., naive_time = 0., omp_time = 0.;
    for (int i = 0; i < ntrials; ++i){
        memset(C1, 0, sizeof(float)*n*n);
        start = omp_get_wtime();
        pald_triplet_naive(D, 1, n, C1);
        naive_time += omp_get_wtime() - start;
    }

    for (int i = 0; i < ntrials; ++i){
        memset(C2, 0, sizeof(float)*n*n);
        start = omp_get_wtime();
        pald_triplet_naive_openmp(D, 1, n, C2, nthreads);
        omp_time += omp_get_wtime() - start;
    }
    
    
    //print out block algorithm result
    // print_out(n, C1);


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
        maxdiff = d > maxdiff ? d : maxdiff;
    }
    // printf("Maximum difference: %1.1e \n", maxdiff);

    printf("=============================================\n");
    printf("           Summary, n: %d\n", n);
    printf("=============================================\n");
    printf("Avg. Sequential time: %.5fs\n",naive_time);
    printf("Avg. Parallel   time: %.5fs, nthreads: %d\n", omp_time, nthreads);
    printf("Speedup: %.2f\n",naive_time/omp_time);
    printf("Parallel Efficiency: %2.2f\n", naive_time/omp_time/nthreads*100);
    printf("Maximum difference: %1.1e\n\n", maxdiff);

    _mm_free(D);
    _mm_free(C2);
    _mm_free(C1);
}