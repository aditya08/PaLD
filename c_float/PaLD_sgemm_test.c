//
// Created by Aditya Devarakonda on 8/17/2022.
//
#include <string.h>
#include "utils.h"
#include "omp.h"
#include "mkl.h"

int main(int argc, char **argv) {

    //initializing testing environment spec
    int n, triplet_cache_size, allz_cache_size, i;
    
    if ((argc < 2) || !(n = atoi(argv[1]))) {
        fprintf(stderr, "Usage: ./name matrix_size\n");
        exit(-1);
    }

    // cache_size = argc > 2 ? 2 : atoi(argv[2]);
    // cache_size = argc == 2 ? 2 : atoi(argv[2]);

    unsigned int num_gen = n * n;

    float *C = _mm_malloc(num_gen*sizeof(float), VECALIGN);
    memset(C, 0, sizeof(float)*num_gen);
    float *A = _mm_malloc(sizeof(float) * num_gen, VECALIGN);
    float *B = _mm_malloc(sizeof(float) * num_gen, VECALIGN);
    sgemm_rand(A, num_gen, -1.f, 1.f, 42);
    sgemm_rand(B, num_gen, -1.f, 1.f, 42);

    int ntrials = 5;
    double start = 0.;


    double sgemm_time = 0.;
    for(int i = 0; i < ntrials; ++i){
        memset(C, 0, sizeof(float)*num_gen);
        start = omp_get_wtime();
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.f, A, n, B, n, 0.f, C, n);
        sgemm_time += omp_get_wtime() - start;
    }
   
    printf("SGEMM time: %.5fs\n", sgemm_time/ntrials);
    //free sgemm matrices.
    _mm_free(A);
    _mm_free(B);
    _mm_free(C);

}
