#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include "kernels.h"
#include "utils.h"

int main(int argc, char **argv) {

    //initializing testing environment spec
    int n, seq_block_size, omp_block_size, t, i;

    // initialize timers
    double start = 0.0, sum = 0.0, time_seq = 0.0 , time_par = 0.0;
    int ntrials = 5;
    if ((argc != 5) || !(n = atoi(argv[1])) || !(seq_block_size = atoi(argv[2])) || !(omp_block_size = atoi(argv[3])) || !(t = atoi(argv[4])) ) {
        fprintf(stderr, "Usage: ./name mat_dim seq_block_size openmp_block_size num_threads\n");
        exit(-1);
    }
    unsigned int num_gen = n * n;

    float *C1 = (float *) _mm_malloc(num_gen*sizeof(float), VECALIGN);
    float *D = (float *) _mm_malloc(num_gen*sizeof(float), VECALIGN);
    float *C2 = (float *) _mm_malloc(num_gen*sizeof(float), VECALIGN);

    dist_mat_gen2D(D, n, 1, 10*n, 12345, '2');

    //computing C with parallel algorithm
    time_par = sum / ntrials;
    //computing C with sequential alg
    sum = 0.;
    for (int i = 0; i < ntrials; i++){
        memset(C1, 0, num_gen*sizeof(float));
        start = omp_get_wtime();
        //pald_allz_orig(D, 1, n, C1);
        pald_allz(D, 1, n, C1, seq_block_size);
        time_seq = omp_get_wtime() - start;
        sum += time_seq;
    }
    time_seq = sum / ntrials;
    sum = 0.;
    for (int i = 0; i < ntrials; ++i){
        memset(C2, 0, num_gen*sizeof(float));
        start = omp_get_wtime();
        //pald_allz_orig_openmp(D, 1, n, C2, omp_block_size, t);
        pald_allz_openmp(D, 1, n, C2, omp_block_size, t);
        time_par = omp_get_wtime() - start;
        sum += time_par;
    }
    // compute max norm error between two cohesion matrices
    float d, maxdiff = 0.;
    for (i = 0; i < num_gen; i++) {
        d = fabs(C1[i]-C2[i]);
        maxdiff = d > maxdiff ? d : maxdiff;
    }

    printf("=============================================\n");
    printf("           Summary, n: %d\n", n);
    printf("=============================================\n");
    printf("Avg. Sequential time: %.5fs\n",time_seq);
    printf("Avg. Parallel   time: %.5fs, nthreads: %d\n", time_par, t);
    printf("Speedup: %.2f\n",time_seq/time_par);
    printf("Parallel Efficiency: %2.2f\n", time_seq/time_par/t*100);
    printf("Maximum difference: %1.1e\n\n", maxdiff);

    _mm_free(D);
    _mm_free(C2);
    _mm_free(C1);
}