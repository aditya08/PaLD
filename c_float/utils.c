//
// Created by Yixin Zhang on 1/10/2021.
//

#include "utils.h"

/*
This is a distance matrix generator

params
D           out empty pointer for the distance matrix:
                D(x,y) is distance between x and y (symmetric,
                but assumed to be stored in full)
edge_len    in  the edge length of a square matrix (total size edge_len^2)
upper_limit in  maximum distance to be generated
seed        in  random seed for matrix generation
*/

void sym_mat_gen(float *D, const int edge_len, const int upper_limit, const unsigned int seed) {

    srand(seed);
    int i, j;
    for (i = 0; i < edge_len - 1; i++) {
        for (j = i + 1; j < edge_len; j++) {
            int temp = rand() % upper_limit + 1;
            D[i * edge_len + j] = temp;
            D[j * edge_len + i] = temp;
        }
    }
    for (i = 0; i < edge_len; i++)
        D[i * edge_len + i] = 0;

}

void point_gen2D(float* x, float* y, int n, int min, int max) {
    for (int i = 0; i < n; i++) {
        x[i] = rand() % (max-min) + min;
        y[i] = rand() % (max-min) + min;
    }
}

void dist_mat_gen2D(float *D, int n, int min, int max, int seed, char dist) {
    srand(seed);
    int i, j;

    // allocate space for points
    float* x = calloc(n, sizeof(float));
    float* y = calloc(n, sizeof(float));

    // generate points in [min,max] x [min,max]
    point_gen2D(x,y,n,min,max);

    //DEBUG
    /*for(i = 0; i < n; i++) {
        printf("%f,%f\n",x[i],y[i]);
    }*/

    // compute pairwise  distances
    for (i = 0; i < n; i++) {
        for (j = i; j < n; j++) {
            if(dist == '1') {
                // L1 distance
                D[i * n + j] = fabs(x[i] - x[j]) + fabs(y[i] - y[j]);
                
            } else if(dist == '2') {
                // L2 distance
                D[i * n + j] = sqrt(pow(x[i] - x[j],2) + pow(y[i] - y[j],2));

            } else {
                printf("Unknown distance!\n");
                exit(1);
            }
            // store explicit symmetry
            D[j * n + i] = D[i * n + j];
        }
    }

    // free up points
    free(y);
    free(x);
}


void sgemm_rand(float *A, int len, float min, float max, int seed){
    srand(seed);
    float fl = -2.f, cl = -2.f;
    for(int i = 0; i < len; ++i){
        A[i] = (float)rand() / RAND_MAX * (max - min) + min;
        cl = (cl > A[i]) ? cl : A[i];
        fl = (fl > A[i]) ? A[i]: fl;
    }
}