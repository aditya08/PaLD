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
    int n, block_size, nthreads, i;
    
    if ((argc != 3 && argc != 4) || !(n = atoi(argv[1]))) {
        fprintf(stderr, "Usage: ./name distance_mat_size block_size num_threads\n");
        exit(-1);
    }

    block_size = atoi(argv[2]);
    nthreads = atoi(argv[3]);
    unsigned int num_gen = n * n;

    float *C1 = (float *) _mm_malloc(num_gen*sizeof(float),VECALIGN);//calloc(num_gen, sizeof(float));
    //float *C2 = calloc(num_gen, sizeof(float));
    float *D = (float *) _mm_malloc(num_gen*sizeof(float),VECALIGN);
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
    int ntrials = 5;
    double start = 0., elapsed = 0.;
    //computing C with optimal block algorithm
    //for (int i = 0; i < 4; ++i)
    for(i = 0; i < ntrials; ++i){
        // pald_allz_openmp(D, 1, n, C1, block_size,nthreads);
        memset(C1, 0, num_gen*sizeof(float));
        start = omp_get_wtime();
        pald_allz(D, 1, n, C1, block_size);
        elapsed += omp_get_wtime() - start;

    }
    
    //print out block algorithm result
    //print_out(n, C);


    //computing C with original algorithm  
    /*   
    start = clock();
    //for (int i = 0; i < 4; ++i)
    pald_orig(D, 1, n, C2);
    diff = clock() - start;
    double msec_orig = 1. * diff / CLOCKS_PER_SEC;
    */

    //print out result of original algorithm
    //print_out(n, C);
    // print out for error checking

    // compute max norm error between two cohesion matrices
    /*
    float d, maxdiff = 0.;
    for (i = 0; i < num_gen; i++) {
        d = fabs(C1[i]-C2[i]);
        maxdiff = d > maxdiff ? d : maxdiff;
    }
    printf("Maximum difference: %1.1e \n", maxdiff);
    */
    printf("%d Opt time: %.3fs\n", n, elapsed/ntrials);
   
    _mm_free(D);
    //free(C2);
    _mm_free(C1);
}
