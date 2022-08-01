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
    int n, cache_size, i;
    
    if ((argc != 2 && argc != 3) || !(n = atoi(argv[1]))) {
        fprintf(stderr, "Usage: ./name distance_mat_size block_size\n");
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
    double start = 0., naive_time = 0.;
    for (int i = 0; i < ntrials; ++i){
        memset(C1, 0, sizeof(float)*n*n);
        start = omp_get_wtime();
        pald_allz(D, 1, n, C1, cache_size);
        naive_time += omp_get_wtime() - start;
    }
    // print_out(n,C1);
    double opt_time = 0.;
    for (int i = 0; i < ntrials; ++i){
        memset(C2, 0, sizeof(float)*n*n);
        start = omp_get_wtime();
        pald_triplet(D, 1, n, C2, cache_size);
        opt_time += omp_get_wtime() - start;
    }
    // print_out(n,C2);
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
    printf("=============================================\n");
    printf("           Summary, n: %d\n", n);
    printf("=============================================\n");
    printf("Triplet Naive Blocked time: %.5fs\n",naive_time/ntrials);
    printf("Triplet Optimized time: %.5fs\n",opt_time/ntrials);

    printf("Speedup: %.2f\n", naive_time/opt_time);
    printf("Maximum difference: %1.8e\n\n", maxdiff);
   
    _mm_free(D);
    _mm_free(C2);
    _mm_free(C1);
}
