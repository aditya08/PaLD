//
// Created by Yixin Zhang on 1/10/2021.
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
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            temp = j * n + i;
            C[temp] /= (n - 1);
            printf("%.7f ", C[temp]);
        }
        printf("\n");
    }
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
    /*
    for (i = 0; i < num_gen; i++) {

        if (i % n == 0) {
            printf("\n");
        }
        printf("%.2f ", D[i]);
    }*/
    FILE *f = fopen("dist_mat.bin", "wb");
    fwrite(D, sizeof(float), num_gen, f);
    fclose(f);
    //computing C with optimal block algorithm
    clock_t start = clock();
    //for (int i = 0; i < 4; ++i)
    pald_allz(D, 1, n, C1, cache_size);
    clock_t diff = clock() - start;
    double  msec_opt = 1. * diff / CLOCKS_PER_SEC;
    
    //print out block algorithm result
    //print_out(n, C);


    //computing C with original algorithm  
    
    start = clock();
    //for (int i = 0; i < 4; ++i)
    pald_allz_naive(D, 1, n, C2);
    diff = clock() - start;
    double msec_orig = 1. * diff / CLOCKS_PER_SEC;


    //print out result of original algorithm
    //print_out(n, C);
    // print out for error checking

    // compute max norm error between two cohesion matrices
    
    float d, maxdiff = 0.;
    for (i = 0; i < num_gen; i++) {
        d = fabs(C1[i]-C2[i]);
        maxdiff = d > maxdiff ? d : maxdiff;
    }
    printf("Maximum difference: %1.1e \n", maxdiff);

    printf("%d  Orig time: %.3fs  Opt time: %.3fs\n", n, msec_orig, msec_opt);
   
    _mm_free(D);
    _mm_free(C2);
    _mm_free(C1);
}
