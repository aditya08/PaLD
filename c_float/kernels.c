#include "kernels.h"
#include "mkl.h"
#include "immintrin.h"
#include "limits.h"
//#include <advisor-annotate.h>

void print_matrix(int size, int stride, float *C) {
    printf("[\n");
    int i, j;
    register int temp;
    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            temp = j * stride + i;
            printf("%.7f ", C[temp]);
        }
        printf(";\n");
    }
    printf("];\n");


}

void print_matrix_int(int size, int stride, unsigned int *C) {
    printf("[\n");
    int i, j;
    register int temp;
    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            temp = j * stride + i;
            printf("%d ", C[temp]);
        }
        printf(";\n");
    }
    printf("];\n");

}

void print_vector(int size, int stride, float *C) {
    printf("[");
    int i, j;
    register int temp;
    for (j = 0; j < size; j++) {
        temp = j * stride;
        printf("%.7f ", C[temp]);
    }
    printf("];\n");

}

// linear indexing function assuming column major
inline int lin(int i, int j, int n) { return i + j * n; }


/*
params
D    in  distance matrix: D(x,y) is distance between x and y (symmetric)
beta in  conflict focus parameter: z is in focus of (x,y) if
         min(d(z,x),d(z,y)) <= beta * d(x,y)
n    in  number of points
C    out cohesion matrix: C(x,z) is z's support for x
*/
void pald_allz_naive(float *D, float beta, int n, float *C) {
    // input checking
    int nties = 0;
    if (beta < 0)
        fprintf(stderr, "beta must be positive\n");

    // loop over pairs of points x and y (only for x < y)
    for (int x = 0; x < n - 1; x++)
        for (int y = x + 1; y < n; y++) {
            int cfs = 0;                // conflict focus size of x,y
            float dxy = D[lin(x, y, n)]; // distance between x and y

            // loop over all points z to determine conflict focus size
            for (int z = 0; z < n; z++) {
                if (D[lin(z, x, n)] <= beta * dxy || D[lin(z, y, n)] <= beta * dxy)
                    cfs++;
            }

            // loop over all points z to determine contributions to x or y
            for (int z = 0; z < n; z++) {
                float dzx = D[lin(z, x, n)]; // dist between z and x
                float dzy = D[lin(z, y, n)]; // dist between z and y

                // z contributes to x or y only if in conflict focus
                if (dzx <= beta * dxy || dzy <= beta * dxy) {
                    if (dzx < dzy)
                        C[lin(x, z, n)] += 1.0f / cfs; // z closer to x than y
                    else if (dzy < dzx)
                        C[lin(y, z, n)] += 1.0f / cfs; // z closer to y than x

                    else {
                        nties++;
                        // z equidistant to x and y
                        C[lin(x, z, n)] += 0.5f / cfs;
                        C[lin(y, z, n)] += 0.5f / cfs;
                    }
                }
            }
        }
    printf("nties: %d\n", nties);
}

/*
params
D    in  distance matrix: D(x,y) is distance between x and y (symmetric,
         but assumed to be stored in full)
beta in  conflict focus parameter: z is in focus of (x,y) if
         min(d(z,x),d(z,y)) <= beta * d(x,y)
n    in  number of points
C    out cohesion matrix: C(x,z) is z's support for x

Optimizations:
Blocking, Masking, Auto-vectorization with  64-byte aligned arrays.
*/
void pald_allz(float* restrict D, float beta, int n, float* restrict C, int block_size) {
    /* TODO: Additional optimization strategies:
    *       1. Handle diagonal blocks separately so that inner loop can be fully vectorized for off-diagonal blocks.
    *       2. Add L1 blocking.
    *       3. Handle non-powers of two input dimensions using another method/set of nested loop to handle the ``remainder''.
    *       4. store conflict_block buffer as an int* type for faster increments -- but this requires one of two things: 1) a second float* buffer to hold 1./conflict_block or 2) on-the-fly floating-point casting and division. Latter is probably very slow.
    *
    */

    // declare indices
    unsigned int x, y, z, i, j, k, xb, yb, ib;
    // pre-allocate buffers for conflict focus and distance blocks
    float * conflict_block = (float *) _mm_malloc(block_size * block_size * sizeof(float),VECALIGN);
    float * distance_block = (float *) _mm_malloc(block_size * block_size * sizeof(float),VECALIGN);
    // pre-allocate mask buffers for z in conflict focus, z support x and z supports both.
    // float * mask_z_in_conflict_focus = (float *) _mm_malloc(block_size * sizeof(float),VECALIGN);
    // float * mask_z_supports_x = (float *) _mm_malloc(block_size  * sizeof(float),VECALIGN);
    // float * mask_z_supports_x_and_y = (float *) _mm_malloc(block_size  * sizeof(float),VECALIGN);

    float z_supports_x, z_supports_y;
    float z_in_conflict_focus;
    float z_supports_x_and_y;
    unsigned int mask_z_in_y_cutoff;
    unsigned int mask_z_in_x_cutoff;

    float CYz_reduction, cutoff_distance;
    float contains_tie = 0.f;
    float CXz_scalar, CYz_scalar;
    // initialize pointers for cache-block subcolumn vectors
    float *CXz;
    float *CYz;
    float *DXz;
    float *DYz;

    unsigned int y_block, x_block, offset;
    float dist_cutoff = 0., dist_cutoff_tmp = 0.;
    double time_start = 0.0;
    double memops_loop_time = 0.0, conflict_loop_time = 0.0, cohesion_loop_time = 0.0;
    //up_left main block

    for (y = 0; y < n; y += block_size) {
        // loop over blocks of points X = (x,...,x+b-1)
        y_block = (block_size < n - y ? block_size : n - y);
        for (x = 0; x <= y; x += block_size) {
            time_start = omp_get_wtime();
            x_block = (block_size < n - x ? block_size : n - x);
            for (j = 0; j < y_block; ++j) {
                // distance_block(:,j) = D(x:x+xb,y+j) in off-diagonal case
                ib = (x == y ? j : x_block); // handle diagonal blocks
                memcpy(distance_block + j * x_block, D + x + (y + j) * n, ib * sizeof(float));
            }

            // compute block's conflict focus sizes by looping over all points z
            memset(conflict_block, 0, block_size * block_size * sizeof(float)); // clear old values

            memops_loop_time += omp_get_wtime() - time_start;
            time_start = omp_get_wtime();
            DXz = D + x;
            DYz = D + y; // init pointers to subcolumns of D
            for (z = 0; z < n; ++z) {
                for (j = 0; j < y_block; ++j) {
                    ib = (x == y ? j : x_block);
                    for (i = 0; i < ib; ++i) {
                        if (DYz[j] <= beta * distance_block[i + j * x_block] || DXz[i] <= beta * distance_block[i + j * x_block]){
                            ++conflict_block[i + j * x_block];
                        }

                    }
                }

                // update pointers to subcolumns of D
                DXz += n;
                DYz += n;
            }
            for (k = 0; k < block_size*block_size; ++k)
                conflict_block[k] = 1.0f/conflict_block[k];
            // print_matrix(n, n, conflict_block);
            conflict_loop_time += omp_get_wtime() - time_start;
            time_start = omp_get_wtime();
            // update cohesion values according to conflicts between X and Y
            // by looping over all points z
            DXz = D + x;
            DYz = D + y; // init pointers to subcolumns of D
            CXz = C + x;
            CYz = C + y; // init pointers to subcolumns of C

            for (z = 0; z < n; ++z) {
                // loop over all (i,j) pairs in block

                for (j = 0; j < y_block; ++j) {
                    ib = (x == y ? j : x_block);
                    CYz_reduction = 0.f;
                    for (i = 0; i < ib; ++i) {
                        z_supports_x = DXz[i] < DYz[j];
                        z_supports_y = DXz[i] > DYz[j];
                        z_supports_x_and_y = (DXz[i] == DYz[j]) ? 0.5f : 0.0f;

                        cutoff_distance = beta*distance_block[i + j * x_block];
                        mask_z_in_x_cutoff = (DXz[i] <= cutoff_distance);
                        mask_z_in_y_cutoff = (DYz[j] <= cutoff_distance);
                        z_in_conflict_focus = mask_z_in_y_cutoff | mask_z_in_x_cutoff;

                        CXz[i] +=  conflict_block[i + j * x_block]*z_in_conflict_focus*(z_supports_x + z_supports_x_and_y);
                        CYz_reduction +=  conflict_block[i + j * x_block]*z_in_conflict_focus*(z_supports_y + z_supports_x_and_y);

                    }
                    CYz[j] += CYz_reduction;
                }

                // update pointers to subcolumns of D and C
                DXz += n;
                DYz += n;
                CXz += n;
                CYz += n;
            }
            cohesion_loop_time += omp_get_wtime() - time_start;
        }
    }

    // printf("nties = %f\n", contains_tie);

    printf("=========================================\n");
    printf("Sequential Allz Optimized Loop Times\n");
    printf("=========================================\n");

    printf("memops loop time: %.5fs\n", memops_loop_time);
    printf("conflict focus size loop time: %.5fs\n", conflict_loop_time);
    printf("cohesion matrix update loop time: %.5fs\n\n", cohesion_loop_time);

    // free up cache blocks
    _mm_free(distance_block);
    _mm_free(conflict_block);
}



/*
params
D    in  distance matrix: D(x,y) is distance between x and y (symmetric,
         but assumed to be stored in full)
beta in  conflict focus parameter: z is in focus of (x,y) if
         min(d(z,x),d(z,y)) <= beta * d(x,y)
n    in  number of points
C    out cohesion matrix: C(x,z) is z's support for x

Optimizations:
Blocking, Masking, Auto-vectorization, no ties and with  64-byte aligned arrays.
*/
void pald_allz_noties(float* restrict D, float beta, int n, float* restrict C, int block_size) {
    /* TODO: Additional optimization strategies:
    *       1. Handle diagonal blocks separately so that inner loop can be fully vectorized for off-diagonal blocks.
    *       2. Add L1 blocking.
    *       3. Handle non-powers of two input dimensions using another method/set of nested loop to handle the ``remainder''.
    *       4. store conflict_block buffer as an int* type for faster increments -- but this requires one of two things: 1) a second float* buffer to hold 1./conflict_block or 2) on-the-fly floating-point casting and division. Latter is probably very slow.
    *
    */

    // declare indices
    unsigned int x, y, z, i, j, k, xb, yb, ib;
    // pre-allocate buffers for conflict focus and distance blocks
    float * conflict_block = (float *) _mm_malloc(block_size * block_size * sizeof(float),VECALIGN);
    float * distance_block = (float *) _mm_malloc(block_size * block_size * sizeof(float),VECALIGN);

    float z_supports_x, z_supports_y;
    float z_in_conflict_focus;
    float z_supports_x_and_y;
    unsigned int mask_z_in_y_cutoff;
    unsigned int mask_z_in_x_cutoff;

    float CYz_reduction, cutoff_distance;
    // initialize pointers for cache-block subcolumn vectors
    float *CXz;
    float *CYz;
    float *DXz;
    float *DYz;

    unsigned int y_block, x_block, offset;
    float dist_cutoff = 0., dist_cutoff_tmp = 0.;
    double time_start = 0.0;
    double memops_loop_time = 0.0, conflict_loop_time = 0.0, cohesion_loop_time = 0.0;
    //up_left main block

    for (y = 0; y < n; y += block_size) {
        // loop over blocks of points X = (x,...,x+b-1)
        y_block = (block_size < n - y ? block_size : n - y);
        for (x = 0; x <= y; x += block_size) {
            time_start = omp_get_wtime();
            x_block = (block_size < n - x ? block_size : n - x);
            for (j = 0; j < y_block; ++j) {
                // distance_block(:,j) = D(x:x+xb,y+j) in off-diagonal case
                ib = (x == y ? j : x_block); // handle diagonal blocks
                memcpy(distance_block + j * x_block, D + x + (y + j) * n, ib * sizeof(float));
            }

            // compute block's conflict focus sizes by looping over all points z
            memset(conflict_block, 0, block_size * block_size * sizeof(float)); // clear old values

            memops_loop_time += omp_get_wtime() - time_start;
            time_start = omp_get_wtime();
            DXz = D + x;
            DYz = D + y; // init pointers to subcolumns of D
            for (z = 0; z < n; ++z) {
                for (j = 0; j < y_block; ++j) {
                    ib = (x == y ? j : x_block);
                    for (i = 0; i < ib; ++i) {
                        if (DYz[j] <= beta * distance_block[i + j * x_block] || DXz[i] <= beta * distance_block[i + j * x_block]){
                            ++conflict_block[i + j * x_block];
                        }

                    }
                }

                // update pointers to subcolumns of D
                DXz += n;
                DYz += n;
            }
            for (k = 0; k < block_size*block_size; ++k)
                conflict_block[k] = 1.0f/conflict_block[k];
            // print_matrix(n, n, conflict_block);
            conflict_loop_time += omp_get_wtime() - time_start;
            time_start = omp_get_wtime();
            // update cohesion values according to conflicts between X and Y
            // by looping over all points z
            DXz = D + x;
            DYz = D + y; // init pointers to subcolumns of D
            CXz = C + x;
            CYz = C + y; // init pointers to subcolumns of C

            for (z = 0; z < n; ++z) {
                // loop over all (i,j) pairs in block

                for (j = 0; j < y_block; ++j) {
                    ib = (x == y ? j : x_block);
                    CYz_reduction = 0.f;
                    for (i = 0; i < ib; ++i) {
                        z_supports_x = DXz[i] < DYz[j];
                        z_supports_y = DXz[i] > DYz[j];

                        cutoff_distance = beta*distance_block[i + j * x_block];
                        mask_z_in_x_cutoff = (DXz[i] <= cutoff_distance);
                        mask_z_in_y_cutoff = (DYz[j] <= cutoff_distance);
                        z_in_conflict_focus = mask_z_in_y_cutoff | mask_z_in_x_cutoff;

                        CXz[i] +=  conflict_block[i + j * x_block]*z_in_conflict_focus*(z_supports_x);
                        CYz_reduction +=  conflict_block[i + j * x_block]*z_in_conflict_focus*(z_supports_y);

                    }
                    CYz[j] += CYz_reduction;
                }

                // update pointers to subcolumns of D and C
                DXz += n;
                DYz += n;
                CXz += n;
                CYz += n;
            }
            cohesion_loop_time += omp_get_wtime() - time_start;
        }
    }

    // printf("nties = %f\n", contains_tie);

    printf("=============================================================\n");
    printf("Sequential Allz blocked + vectorized + noties Loop Times\n");
    printf("=============================================================\n");

    printf("memops loop time: %.5fs\n", memops_loop_time);
    printf("conflict focus size loop time: %.5fs\n", conflict_loop_time);
    printf("cohesion matrix update loop time: %.5fs\n\n", cohesion_loop_time);

    // free up cache blocks
    _mm_free(distance_block);
    _mm_free(conflict_block);
}


/*
params
D    in  distance matrix: D(x,y) is distance between x and y (symmetric,
         but assumed to be stored in full)
beta in  conflict focus parameter: z is in focus of (x,y) if
         min(d(z,x),d(z,y)) <= beta * d(x,y)
n    in  number of points
C    out cohesion matrix: C(x,z) is z's support for x

Optimizations:
Blocking, Masking, Auto-vectorization, no ties and with  64-byte aligned arrays.
*/
void pald_allz_noties_nobeta(float* restrict D, float beta, int n, float* restrict C, int block_size) {
    /* TODO: Additional optimization strategies:
    *       1. Handle diagonal blocks separately so that inner loop can be fully vectorized for off-diagonal blocks.
    *       2. Add L1 blocking.
    *       3. Handle non-powers of two input dimensions using another method/set of nested loop to handle the ``remainder''.
    *       4. store conflict_block buffer as an int* type for faster increments -- but this requires one of two things: 1) a second float* buffer to hold 1./conflict_block or 2) on-the-fly floating-point casting and division. Latter is probably very slow.
    *
    */

    // declare indices
    unsigned int x, y, z, i, j, k, xb, yb, ib;
    // pre-allocate buffers for conflict focus and distance blocks
    float * conflict_block = (float *) _mm_malloc(block_size * block_size * sizeof(float),VECALIGN);
    float * distance_block = (float *) _mm_malloc(block_size * block_size * sizeof(float),VECALIGN);

    float z_supports_x, z_supports_y;
    float z_in_conflict_focus;
    unsigned int mask_z_in_y_cutoff;
    unsigned int mask_z_in_x_cutoff;

    // float cutoff_distance;
    float CYz_reduction;
    // initialize pointers for cache-block subcolumn vectors
    float *CXz;
    float *CYz;
    float *DXz;
    float *DYz;

    unsigned int y_block, x_block;
    float dist_cutoff = 0.;
    double time_start = 0.0;
    double memops_loop_time = 0.0, conflict_loop_time = 0.0, cohesion_loop_time = 0.0;
    //up_left main block

    for (y = 0; y < n; y += block_size) {
        // loop over blocks of points X = (x,...,x+b-1)
        y_block = (block_size < n - y ? block_size : n - y);
        for (x = 0; x <= y; x += block_size) {
            time_start = omp_get_wtime();
            x_block = (block_size < n - x ? block_size : n - x);
            for (j = 0; j < y_block; ++j) {
                // distance_block(:,j) = D(x:x+xb,y+j) in off-diagonal case
                ib = (x == y ? j : x_block); // handle diagonal blocks
                memcpy(distance_block + j * x_block, D + x + (y + j) * n, ib * sizeof(float));
            }

            // compute block's conflict focus sizes by looping over all points z
            memset(conflict_block, 0, block_size * block_size * sizeof(float)); // clear old values

            memops_loop_time += omp_get_wtime() - time_start;
            time_start = omp_get_wtime();
            DXz = D + x;
            DYz = D + y; // init pointers to subcolumns of D
            for (z = 0; z < n; ++z) {
                for (j = 0; j < y_block; ++j) {
                    ib = (x == y ? j : x_block);
                    for (i = 0; i < ib; ++i) {
                        // if (DYz[j] <= beta * distance_block[i + j * x_block] || DXz[i] <= beta * distance_block[i + j * x_block]){
                        if (DYz[j] <= distance_block[i + j * x_block] || DXz[i] <= distance_block[i + j * x_block]){
                            ++conflict_block[i + j * x_block];
                        }

                    }
                }

                // update pointers to subcolumns of D
                DXz += n;
                DYz += n;
            }
            for (k = 0; k < block_size*block_size; ++k)
                conflict_block[k] = 1.0f/conflict_block[k];
            // print_matrix(n, n, conflict_block);
            conflict_loop_time += omp_get_wtime() - time_start;
            time_start = omp_get_wtime();
            // update cohesion values according to conflicts between X and Y
            // by looping over all points z
            DXz = D + x;
            DYz = D + y; // init pointers to subcolumns of D
            CXz = C + x;
            CYz = C + y; // init pointers to subcolumns of C

            for (z = 0; z < n; ++z) {
                // loop over all (i,j) pairs in block

                for (j = 0; j < y_block; ++j) {
                    ib = (x == y ? j : x_block);
                    CYz_reduction = 0.f;
                    for (i = 0; i < ib; ++i) {
                        z_supports_x = DXz[i] < DYz[j];
                        z_supports_y = DXz[i] > DYz[j];

                        // cutoff_distance = beta*distance_block[i + j * x_block];
                        // mask_z_in_x_cutoff = (DXz[i] <= cutoff_distance);
                        // mask_z_in_y_cutoff = (DYz[j] <= cutoff_distance);
                        mask_z_in_x_cutoff = (DXz[i] <= distance_block[i + j * x_block]);
                        mask_z_in_y_cutoff = (DYz[j] <= distance_block[i + j * x_block]);
                        z_in_conflict_focus = mask_z_in_y_cutoff | mask_z_in_x_cutoff;

                        CXz[i] +=  conflict_block[i + j * x_block]*z_in_conflict_focus*(z_supports_x);
                        CYz_reduction +=  conflict_block[i + j * x_block]*z_in_conflict_focus*(z_supports_y);

                    }
                    CYz[j] += CYz_reduction;
                }

                // update pointers to subcolumns of D and C
                DXz += n;
                DYz += n;
                CXz += n;
                CYz += n;
            }
            cohesion_loop_time += omp_get_wtime() - time_start;
        }
    }

    // printf("nties = %f\n", contains_tie);

    printf("======================================================================\n");
    printf("Sequential Allz blocked + vectorized + noties + nobeta Loop Times\n");
    printf("======================================================================\n");

    printf("memops loop time: %.5fs\n", memops_loop_time);
    printf("conflict focus size loop time: %.5fs\n", conflict_loop_time);
    printf("cohesion matrix update loop time: %.5fs\n\n", cohesion_loop_time);

    // free up cache blocks
    _mm_free(distance_block);
    _mm_free(conflict_block);
}


/*
params
D    in  distance matrix: D(x,y) is distance between x and y (symmetric,
         but assumed to be stored in full)
beta in  conflict focus parameter: z is in focus of (x,y) if
         min(d(z,x),d(z,y)) <= beta * d(x,y)
n    in  number of points
C    out cohesion matrix: C(x,z) is z's support for x

Optimizations:
Blocking, Masking, Auto-vectorization, no ties, no beta, and with branched vec loops, 64-byte aligned arrays.
*/
void pald_allz_noties_nobeta_vecbranching(float* restrict D, float beta, int n, float* restrict C, int block_size) {
    /* TODO: Additional optimization strategies:
    *       1. Handle diagonal blocks separately so that inner loop can be fully vectorized for off-diagonal blocks.
    *       2. Add L1 blocking.
    *       3. Handle non-powers of two input dimensions using another method/set of nested loop to handle the ``remainder''.
    *       4. store conflict_block buffer as an int* type for faster increments -- but this requires one of two things: 1) a second float* buffer to hold 1./conflict_block or 2) on-the-fly floating-point casting and division. Latter is probably very slow.
    *
    */

    // declare indices
    unsigned int x, y, z, i, j, k, xb, yb, ib;
    // pre-allocate buffers for conflict focus and distance blocks
    float * conflict_block = (float *) _mm_malloc(block_size * block_size * sizeof(float),VECALIGN);
    float * distance_block = (float *) _mm_malloc(block_size * block_size * sizeof(float),VECALIGN);

    float z_supports_x, z_supports_y;
    float z_in_conflict_focus;
    unsigned int mask_z_in_y_cutoff;
    unsigned int mask_z_in_x_cutoff;

    // float cutoff_distance;
    float CYz_reduction;
    // initialize pointers for cache-block subcolumn vectors
    float *CXz;
    float *CYz;
    float *DXz;
    float *DYz;

    unsigned int y_block, x_block;
    float dist_cutoff = 0.;
    double time_start = 0.0;
    double memops_loop_time = 0.0, conflict_loop_time = 0.0, cohesion_loop_time = 0.0;
    //up_left main block

    for (y = 0; y < n; y += block_size) {
        // loop over blocks of points X = (x,...,x+b-1)
        y_block = (block_size < n - y ? block_size : n - y);
        for (x = 0; x <= y; x += block_size) {
            time_start = omp_get_wtime();
            x_block = (block_size < n - x ? block_size : n - x);
            for (j = 0; j < y_block; ++j) {
                // distance_block(:,j) = D(x:x+xb,y+j) in off-diagonal case
                ib = (x == y ? j : x_block); // handle diagonal blocks
                memcpy(distance_block + j * x_block, D + x + (y + j) * n, ib * sizeof(float));
            }

            // compute block's conflict focus sizes by looping over all points z
            memset(conflict_block, 0, block_size * block_size * sizeof(float)); // clear old values

            memops_loop_time += omp_get_wtime() - time_start;
            time_start = omp_get_wtime();
            DXz = D + x;
            DYz = D + y; // init pointers to subcolumns of D
            for (z = 0; z < n; ++z) {
                for (j = 0; j < y_block; ++j) {
                    // ib = (x == y ? j : x_block);
                    if(x == y){
                        for (i = 0; i < j; ++i) {
                        // if (DYz[j] <= beta * distance_block[i + j * x_block] || DXz[i] <= beta * distance_block[i + j * x_block]){
                            if (DYz[j] <= distance_block[i + j * x_block] || DXz[i] <= distance_block[i + j * x_block]){
                                ++conflict_block[i + j * x_block];
                            }
                        }
                    }
                    else{
                        for (i = 0; i < x_block; ++i) {
                            // if (DYz[j] <= beta * distance_block[i + j * x_block] || DXz[i] <= beta * distance_block[i + j * x_block]){
                            if (DYz[j] <= distance_block[i + j * x_block] || DXz[i] <= distance_block[i + j * x_block]){
                                ++conflict_block[i + j * x_block];
                            }

                        }
                    }
                }

                // update pointers to subcolumns of D
                DXz += n;
                DYz += n;
            }
            for (k = 0; k < block_size*block_size; ++k)
                conflict_block[k] = 1.0f/conflict_block[k];
            // print_matrix(n, n, conflict_block);
            conflict_loop_time += omp_get_wtime() - time_start;
            time_start = omp_get_wtime();
            // update cohesion values according to conflicts between X and Y
            // by looping over all points z
            DXz = D + x;
            DYz = D + y; // init pointers to subcolumns of D
            CXz = C + x;
            CYz = C + y; // init pointers to subcolumns of C

            for (z = 0; z < n; ++z) {
                // loop over all (i,j) pairs in block

                for (j = 0; j < y_block; ++j) {
                    // ib = (x == y ? j : x_block);
                    CYz_reduction = 0.f;
                    if(x == y){
                        for (i = 0; i < j; ++i) {
                            z_supports_x = DXz[i] < DYz[j];
                            z_supports_y = DXz[i] > DYz[j];

                            // cutoff_distance = beta*distance_block[i + j * x_block];
                            // mask_z_in_x_cutoff = (DXz[i] <= cutoff_distance);
                            // mask_z_in_y_cutoff = (DYz[j] <= cutoff_distance);
                            mask_z_in_x_cutoff = (DXz[i] <= distance_block[i + j * x_block]);
                            mask_z_in_y_cutoff = (DYz[j] <= distance_block[i + j * x_block]);
                            z_in_conflict_focus = mask_z_in_y_cutoff | mask_z_in_x_cutoff;

                            CXz[i] +=  conflict_block[i + j * x_block]*z_in_conflict_focus*(z_supports_x);
                            CYz_reduction +=  conflict_block[i + j * x_block]*z_in_conflict_focus*(z_supports_y);
                        }
                    }
                    else{
                        for (i = 0; i < x_block; ++i) {
                            z_supports_x = DXz[i] < DYz[j];
                            z_supports_y = DXz[i] > DYz[j];

                            // cutoff_distance = beta*distance_block[i + j * x_block];
                            // mask_z_in_x_cutoff = (DXz[i] <= cutoff_distance);
                            // mask_z_in_y_cutoff = (DYz[j] <= cutoff_distance);
                            mask_z_in_x_cutoff = (DXz[i] <= distance_block[i + j * x_block]);
                            mask_z_in_y_cutoff = (DYz[j] <= distance_block[i + j * x_block]);
                            z_in_conflict_focus = mask_z_in_y_cutoff | mask_z_in_x_cutoff;

                            CXz[i] +=  conflict_block[i + j * x_block]*z_in_conflict_focus*(z_supports_x);
                            CYz_reduction +=  conflict_block[i + j * x_block]*z_in_conflict_focus*(z_supports_y);
                        }
                    }
                    CYz[j] += CYz_reduction;
                }

                // update pointers to subcolumns of D and C
                DXz += n;
                DYz += n;
                CXz += n;
                CYz += n;
            }
            cohesion_loop_time += omp_get_wtime() - time_start;
        }
    }

    // printf("nties = %f\n", contains_tie);

    printf("====================================================================================================\n");
    printf("Sequential Allz blocked + vectorized + noties + nobeta + vectorized loop branching Loop Times\n");
    printf("====================================================================================================\n");

    printf("memops loop time: %.5fs\n", memops_loop_time);
    printf("conflict focus size loop time: %.5fs\n", conflict_loop_time);
    printf("cohesion matrix update loop time: %.5fs\n\n", cohesion_loop_time);

    // free up cache blocks
    _mm_free(distance_block);
    _mm_free(conflict_block);
}

void pald_allz_experimental(float* restrict D, float beta, int n, float* restrict C, int block_size) {
    /* TODO: Additional optimization strategies:
    *       1. Handle diagonal blocks separately so that inner loop can be fully vectorized for off-diagonal blocks.
    *       2. Add L1 blocking.
    *       3. Handle non-powers of two input dimensions using another method/set of nested loop to handle the ``remainder''.
    *       4. store conflict_block buffer as an int* type for faster increments -- but this requires one of two things: 1) a second float* buffer to hold 1./conflict_block or 2) on-the-fly floating-point casting and division. Latter is probably very slow.
    *
    */

    // declare indices
    int x, y, z, i, j, k, xb, yb, ib;
    // pre-allocate buffers for conflict focus and distance blocks
    float * conflict_block = (float *) _mm_malloc(block_size * block_size * sizeof(float),VECALIGN);
    int * conflict_block_int = (int *) _mm_malloc(block_size * block_size * sizeof(int),VECALIGN);
    float * distance_block = (float *) _mm_malloc(block_size * block_size * sizeof(float),VECALIGN);
    // pre-allocate mask buffers for z in conflict focus, z support x and z supports both.
    float * mask_z_in_conflict_focus = (float *) _mm_malloc(block_size * sizeof(float),VECALIGN);
    float * mask_z_supports_x = (float *) _mm_malloc(block_size  * sizeof(float),VECALIGN);
    float * mask_z_supports_x_and_y = (float *) _mm_malloc(block_size  * sizeof(float),VECALIGN);
    __mmask16 distance_check_1, distance_check_2;
    // __mmask16 cmp_result;
    // __m512i all_ones = _mm512_set1_epi32(1);
    // __m512i masked_ones;
    // __m512 beta_avx = _mm512_set1_ps(beta);
    // __m512 dist_yz_avx;
    // __m512 distance_block_avx;
    // __m512 dist_xz_avx;
    // __m512 cutoff_avx;
    // __m512i conflict_block_avx;
    // __m512i all_zeros = _mm512_setzero_epi32();

    unsigned short * mask_conflict = (unsigned short *) _mm_malloc(block_size * sizeof(unsigned short), VECALIGN);
    float dist_yz;
    char mask_z_in_y_cutoff  = 0;
    char mask_z_in_x_cutoff = 0;
    float CYz_reduction, contains_tie, cutoff_distance;

    // initialize pointers for cache-block subcolumn vectors
    float *CXz;
    float *CYz;
    float *DXz;
    float *DYz;

    int y_block, x_block, offset, remainder;
    float dist_cutoff = 0., dist_cutoff_tmp = 0.;
    double time_start = 0.0;
    double memops_loop_time = 0.0, conflict_loop_time = 0.0, cohesion_loop_time = 0.0;
    //up_left main block

    for (y = 0; y < n; y += block_size) {
        // loop over blocks of points X = (x,...,x+b-1)
        y_block = (block_size < n - y ? block_size : n - y);
        for (x = 0; x <= y; x += block_size) {
            time_start = omp_get_wtime();
            x_block = (block_size < n - x ? block_size : n - x);
            for (j = 0; j < y_block; ++j) {
                // distance_block(:,j) = D(x:x+xb,y+j) in off-diagonal case
                ib = (x == y ? j : x_block); // handle diagonal blocks
                memcpy(distance_block + j * x_block, D + x + (y + j) * n, ib * sizeof(float));
            }

            // compute block's conflict focus sizes by looping over all points z
            // memset(conflict_block, 0, block_size * block_size * sizeof(float)); // clear old values
            memset(conflict_block_int, 0, block_size * block_size * sizeof(int)); // clear old values
            //  for(k = 0; k < block_size*block_size; ++k)
            //      conflict_block[k] = 0.;

            memops_loop_time += omp_get_wtime() - time_start;
            time_start = omp_get_wtime();
            DXz = D + x;
            DYz = D + y; // init pointers to subcolumns of D
            for (z = 0; z < n; ++z) {
                for (j = 0; j < y_block; ++j) {
                    dist_yz = DYz[j];
                    ib = (x == y ? j : x_block);
                    // #pragma unroll(16)
                    /*
                    * Need to implement intrinsics to count number of bits set to 1 in cmp_result.
                    * Significant slowdown when you do conflict_block_int[idx] += cmp_result, due to cast op and no vectorization.
                    * Use _mm512_set1_ps to load beta and dist_yz into two 512-registers.
                    * Use _mm512_load_ps to load 16 floats from distance_block into one 512-register.
                    */
                //    if(ib >= 16){
                //         remainder = ib % 16;
                //         dist_yz_avx = _mm512_set1_ps(dist_yz);
                //         for (i = 0; i < ib - remainder; i+=16) {
                //             distance_block_avx = _mm512_load_ps(distance_block + i + j * block_size);
                //             dist_xz_avx = _mm512_load_ps(DXz + i);
                //             conflict_block_avx = _mm512_load_epi32(conflict_block_int + (j * block_size) + i);
                //             cutoff_avx = _mm512_mul_ps(beta_avx, distance_block_avx);
                //             distance_check_1 = _mm512_cmple_ps_mask(dist_yz_avx, cutoff_avx);
                //             distance_check_2 = _mm512_cmple_ps_mask(dist_xz_avx, cutoff_avx);
                //             cmp_result = distance_check_1 | distance_check_2;
                //             conflict_block_avx = _mm512_mask_add_epi32(conflict_block_avx, cmp_result, conflict_block_avx, all_ones);
                //             _mm512_store_epi32(conflict_block_int + (j * block_size) + i, conflict_block_avx);
                //             // distance_check_1 = dist_yz <= beta * distance_block[i + j * x_block];
                //             // distance_check_2 = DXz[i] <= beta * distance_block[i + j * x_block];
                //             // cmp_result = distance_check_1 | distance_check_2;

                //             // conflict_block_int[i + j * x_block] += _mm512_mask_reduce_add_epi32(cmp_result, all_ones);
                //             // conflict_block_int[i + j * x_block] += distance_check_1 | distance_check_2;
                //             // printf("%d, %d\n",_mm512_kor(distance_check_1, distance_check_2), distance_check_1 | distance_check_2);
                //             // if (DYz[j] <= beta * distance_block[i + j * x_block] || DXz[i] <= beta * distance_block[i + j * x_block]){
                //             //     ++conflict_block_int[i + j * x_block];
                //             // }

                //         }
                //         for(i = ib-remainder; i < ib; ++i){
                //             distance_check_1 = dist_yz <= beta * distance_block[i + j * x_block];
                //             distance_check_2 = DXz[i] <= beta * distance_block[i + j * x_block];
                //             conflict_block_int[i + j * x_block] += distance_check_1 | distance_check_2;
                //         }
                //    }
                //    else{

                    for(i = 0; i < ib; ++i){
                        // cutoff_distance = beta * distance_block[i + j * x_block];
                        distance_check_1 = dist_yz <= beta * distance_block[i + j * x_block];
                        distance_check_2 = DXz[i] <= beta * distance_block[i + j * x_block];
                        conflict_block_int[i + j * x_block] += (distance_check_1 | distance_check_2);
                        // if (DYz[j] <= beta * distance_block[i + j * x_block] || DXz[i] <= beta * distance_block[i + j * x_block]){
                        //     ++conflict_block_int[i + j * x_block];
                        // }

                    }
                //    }
                }
                // update pointers to subcolumns of D
                DXz += n;
                DYz += n;
            }
            for (k = 0; k < block_size*block_size; ++k)
                conflict_block[k] = 1.0f/conflict_block_int[k];
            // print_matrix(n, n, conflict_block);
            conflict_loop_time += omp_get_wtime() - time_start;
            time_start = omp_get_wtime();
            // update cohesion values according to conflicts between X and Y
            // by looping over all points z
            DXz = D + x;
            DYz = D + y; // init pointers to subcolumns of D
            CXz = C + x;
            CYz = C + y; // init pointers to subcolumns of C


            for (z = 0; z < n; ++z) {
                // loop over all (i,j) pairs in block
                for (j = 0; j < y_block; ++j) {
                    ib = (x == y ? j : x_block);
                    for (i = 0; i < ib; ++i) {
                        // mask_z_supports_x is when z support x.
                        mask_z_supports_x[i] = DXz[i] < DYz[j];
                        // mask_z_supports_x_and_y[i] = DXz[i] == DYz[j] ? 1.0f : 0.0f;
                        cutoff_distance = beta*distance_block[i + j * x_block];
                        mask_z_in_x_cutoff = (DXz[i] <= cutoff_distance);
                        mask_z_in_y_cutoff = (DYz[j] <= cutoff_distance);
                        mask_z_in_conflict_focus[i] = mask_z_in_y_cutoff | mask_z_in_x_cutoff;
                    }
                    // for (i = 0; i < ib; ++i) {
                    // //     // mask_z_supports_x_and_y is when z supports both x and y. Support should be divided between x and y.
                    //     // contains_tie += mask_z_supports_x_and_y[i];
                    // }
                    CYz_reduction = 0;
                    for (i = 0; i<ib; ++i){
                        CXz[i] +=  conflict_block[i + j * x_block]*mask_z_in_conflict_focus[i]*(mask_z_supports_x[i]);
                        // 1 - mask_z_supports_x ==> z supports y and z supports both.
                        CYz_reduction +=  conflict_block[i + j * x_block]*mask_z_in_conflict_focus[i]*(1 - mask_z_supports_x[i]);
                    }
                    CYz[j] += CYz_reduction;

                    // if (contains_tie > 0.5f){
                    //     contains_tie = 0.f;
                    //     CYz_reduction = CYz[j];
                    //     for (i = 0; i < ib; ++i){
                    //         CXz[i] += 0.5f * conflict_block[i + j * x_block]*mask_z_in_conflict_focus[i]*mask_z_supports_x_and_y[i];
                    //         CYz_reduction -= 0.5f * conflict_block[i + j * x_block]*mask_z_in_conflict_focus[i]*mask_z_supports_x_and_y[i];
                    //         // mask_z_supports_x contains ties so subtract 1/2.
                    //     }
                    //     CYz[j] = CYz_reduction;
                    // }
                }

                // update pointers to subcolumns of D and C
                DXz += n;
                DYz += n;
                CXz += n;
                CYz += n;
            }
            cohesion_loop_time += omp_get_wtime() - time_start;
        }
    }

    printf("=========================================\n");
    printf("Sequential Allz Experimental Loop Times\n");
    printf("=========================================\n");

    printf("memops loop time: %.5fs\n", memops_loop_time);
    printf("conflict focus size loop time: %.5fs\n", conflict_loop_time);
    printf("cohesion matrix update loop time: %.5fs\n\n", cohesion_loop_time);

    // free up cache blocks
    _mm_free(mask_z_in_conflict_focus);
    _mm_free(mask_z_supports_x);
    _mm_free(mask_z_supports_x_and_y);
    _mm_free(distance_block);
    _mm_free(conflict_block);
}

/*
params
D    in  distance matrix: D(x,y) is distance between x and y (symmetric,
         but assumed to be stored in full)
beta in  conflict focus parameter: z is in focus of (x,y) if
         min(d(z,x),d(z,y)) <= beta * d(x,y)
n    in  number of points
C    out cohesion matrix: C(x,z) is z's support for x

Optimizations:
OpenMP, no-ties, Blocking, Masking, Auto-vectorization with  64-byte aligned arrays.
*/

void pald_allz_openmp_noties(float* restrict D, float beta, int n, float* restrict C, int block_size, int nthreads) {
    /* TODO: Additional optimization strategies:
    *       1. Handle diagonal blocks separately so that inner loop can be fully vectorized for off-diagonal blocks.
    *       2. Add L1 blocking.
    *       3. Handle non-powers of two input dimensions using another method/set of nested loop to handle the ``remainder''.
    *       4. store conflict_block buffer as an int* type for faster increments -- but this requires one of two things: 1) a second float* buffer to hold 1./conflict_block or 2) on-the-fly floating-point casting and division. Latter is probably very slow.
    *
    */

    // declare indices
    unsigned int x, y, z, i, j, k, xb, yb, ib;
    // pre-allocate buffers for conflict focus and distance blocks
    float * conflict_block = (float *) _mm_malloc(block_size * block_size * sizeof(float),VECALIGN);
    float * distance_block = (float *) _mm_malloc(block_size * block_size * sizeof(float),VECALIGN);


    // initialize pointers for cache-block subcolumn vectors
    unsigned int y_block, x_block;
    double time_start = 0.0;
    double memops_loop_time = 0.0, conflict_loop_time = 0.0, cohesion_loop_time = 0.0;
    //up_left main block
    for (y = 0; y < n; y += block_size) {
        // loop over blocks of points X = (x,...,x+b-1)
        y_block = (block_size < n - y ? block_size : n - y);
        for (x = 0; x <= y; x += block_size) {
            time_start = omp_get_wtime();
            x_block = (block_size < n - x ? block_size : n - x);
            for (j = 0; j < y_block; ++j) {
                ib = (x == y ? j : x_block); // handle diagonal blocks
                // distance_block(:,j) = D(x:x+xb,y+j) in off-diagonal case
                memcpy(distance_block + j * x_block, D + x + (y + j) * n, ib * sizeof(float));
            }

            // compute block's conflict focus sizes by looping over all points z
            memset(conflict_block, 0, block_size * block_size * sizeof(float)); // clear old values

            memops_loop_time += omp_get_wtime() - time_start;
            time_start = omp_get_wtime();
            #pragma omp parallel for num_threads(nthreads) private(i, j, ib, z) reduction(+:conflict_block[:block_size*block_size]) schedule(static)
            for (z = 0; z < n; ++z) {
                float* DXz = D + x + z*n;
                float* DYz = D + y + z*n;
                for (j = 0; j < y_block; ++j) {
                    ib = (x == y ? j : x_block);
                    for (i = 0; i < ib; ++i) {
                        if (DYz[j] <= beta * distance_block[i + j * x_block] || DXz[i] <= beta * distance_block[i + j * x_block]){
                            ++conflict_block[i + j * x_block];
                        }

                    }
                }
            }
            #pragma omp parallel for num_threads(nthreads)
            for (k = 0; k < block_size*block_size; ++k)
                conflict_block[k] = 1.0f/conflict_block[k];
            // print_matrix(n, n, conflict_block);
            conflict_loop_time += omp_get_wtime() - time_start;
            time_start = omp_get_wtime();
            #pragma omp parallel num_threads(nthreads) private(i, j, ib, z)
            {
                float *CXz, *CYz, *DXz, *DYz;
                float z_supports_x;
                float z_supports_y;
                float z_supports_x_and_y;
                float z_in_conflict_focus;
                unsigned int mask_z_in_y_cutoff;
                unsigned int mask_z_in_x_cutoff;

                float CYz_reduction = 0.f;
                float cutoff_distance = 0.f;

                #pragma omp for nowait schedule(static)
                for (z = 0; z < n; ++z) {
                    // loop over all (i,j) pairs in block
                    DXz = D + x + z*n;
                    DYz = D + y + z*n;
                    CXz = C + x + z*n;
                    CYz = C + y + z*n;
                    for (j = 0; j < y_block; ++j) {
                        ib = (x == y ? j : x_block);
                        CYz_reduction = 0.f;
                        for (i = 0; i < ib; ++i) {
                            z_supports_x = DXz[i] < DYz[j];
                            z_supports_y = DXz[i] > DYz[j];

                            cutoff_distance = beta*distance_block[i + j * x_block];
                            mask_z_in_x_cutoff = (DXz[i] <= cutoff_distance);
                            mask_z_in_y_cutoff = (DYz[j] <= cutoff_distance);
                            z_in_conflict_focus = mask_z_in_y_cutoff | mask_z_in_x_cutoff;

                            CXz[i] +=  conflict_block[i + j * x_block]*z_in_conflict_focus*(z_supports_x);
                            CYz_reduction +=  conflict_block[i + j * x_block]*z_in_conflict_focus*(z_supports_y);

                        }
                        CYz[j] += CYz_reduction;
                    }
                }
            }
            cohesion_loop_time += omp_get_wtime() - time_start;
        }
    }

    // printf("nties = %f\n", contains_tie);

    printf("=========================================\n");
    printf("Allz OpenMP noties Loop Times\n");
    printf("=========================================\n");

    printf("memops loop time: %.5fs\n", memops_loop_time);
    printf("conflict focus size loop time: %.5fs\n", conflict_loop_time);
    printf("cohesion matrix update loop time: %.5fs\n\n", cohesion_loop_time);

    // free up cache blocks
    _mm_free(distance_block);
    _mm_free(conflict_block);
}

/*
params
D    in  distance matrix: D(x,y) is distance between x and y (symmetric,
         but assumed to be stored in full)
beta in  conflict focus parameter: z is in focus of (x,y) if
         min(d(z,x),d(z,y)) <= beta * d(x,y)
n    in  number of points
C    out cohesion matrix: C(x,z) is z's support for x

Optimizations:
OpenMP, no-ties, nobeta, Blocking, Masking, Auto-vectorization with  64-byte aligned arrays.
*/

void pald_allz_openmp_noties_nobeta(float* restrict D, float beta, int n, float* restrict C, int block_size, int nthreads) {
    /* TODO: Additional optimization strategies:
    *       1. Handle diagonal blocks separately so that inner loop can be fully vectorized for off-diagonal blocks.
    *       2. Add L1 blocking.
    *       3. Handle non-powers of two input dimensions using another method/set of nested loop to handle the ``remainder''.
    *       4. store conflict_block buffer as an int* type for faster increments -- but this requires one of two things: 1) a second float* buffer to hold 1./conflict_block or 2) on-the-fly floating-point casting and division. Latter is probably very slow.
    *
    */

    // declare indices
    unsigned int x, y, z, i, j, k, xb, yb, ib;
    // pre-allocate buffers for conflict focus and distance blocks
    float * conflict_block = (float *) _mm_malloc(block_size * block_size * sizeof(float),VECALIGN);
    float * distance_block = (float *) _mm_malloc(block_size * block_size * sizeof(float),VECALIGN);


    // initialize pointers for cache-block subcolumn vectors
    unsigned int y_block, x_block;
    double time_start = 0.0;
    double memops_loop_time = 0.0, conflict_loop_time = 0.0, cohesion_loop_time = 0.0;
    //up_left main block
    for (y = 0; y < n; y += block_size) {
        // loop over blocks of points X = (x,...,x+b-1)
        y_block = (block_size < n - y ? block_size : n - y);
        for (x = 0; x <= y; x += block_size) {
            time_start = omp_get_wtime();
            x_block = (block_size < n - x ? block_size : n - x);
            for (j = 0; j < y_block; ++j) {
                ib = (x == y ? j : x_block); // handle diagonal blocks
                // distance_block(:,j) = D(x:x+xb,y+j) in off-diagonal case
                memcpy(distance_block + j * x_block, D + x + (y + j) * n, ib * sizeof(float));
            }

            // compute block's conflict focus sizes by looping over all points z
            memset(conflict_block, 0, block_size * block_size * sizeof(float)); // clear old values

            memops_loop_time += omp_get_wtime() - time_start;
            time_start = omp_get_wtime();
            #pragma omp parallel for num_threads(nthreads) private(i, j, ib, z) reduction(+:conflict_block[:block_size*block_size]) schedule(static)
            for (z = 0; z < n; ++z) {
                float* DXz = D + x + z*n;
                float* DYz = D + y + z*n;
                for (j = 0; j < y_block; ++j) {
                    ib = (x == y ? j : x_block);
                    for (i = 0; i < ib; ++i) {
                        if (DYz[j] <= distance_block[i + j * x_block] || DXz[i] <= distance_block[i + j * x_block]){
                            ++conflict_block[i + j * x_block];
                        }

                    }
                }
            }
            #pragma omp parallel for num_threads(nthreads)
            for (k = 0; k < block_size*block_size; ++k)
                conflict_block[k] = 1.0f/conflict_block[k];
            // print_matrix(n, n, conflict_block);
            conflict_loop_time += omp_get_wtime() - time_start;
            time_start = omp_get_wtime();
            #pragma omp parallel num_threads(nthreads) private(i, j, ib, z)
            {
                float *CXz, *CYz, *DXz, *DYz;
                float z_supports_x;
                float z_supports_y;
                float z_supports_x_and_y;
                float z_in_conflict_focus;
                unsigned int mask_z_in_y_cutoff;
                unsigned int mask_z_in_x_cutoff;

                float CYz_reduction = 0.f;

                #pragma omp for nowait schedule(static)
                for (z = 0; z < n; ++z) {
                    // loop over all (i,j) pairs in block
                    DXz = D + x + z*n;
                    DYz = D + y + z*n;
                    CXz = C + x + z*n;
                    CYz = C + y + z*n;
                    for (j = 0; j < y_block; ++j) {
                        ib = (x == y ? j : x_block);
                        CYz_reduction = 0.f;
                        for (i = 0; i < ib; ++i) {
                            z_supports_x = DXz[i] < DYz[j];
                            z_supports_y = DXz[i] > DYz[j];

                            mask_z_in_x_cutoff = (DXz[i] <=  distance_block[i + j * x_block]);
                            mask_z_in_y_cutoff = (DYz[j] <=  distance_block[i + j * x_block]);
                            z_in_conflict_focus = mask_z_in_y_cutoff | mask_z_in_x_cutoff;

                            CXz[i] +=  conflict_block[i + j * x_block]*z_in_conflict_focus*(z_supports_x);
                            CYz_reduction +=  conflict_block[i + j * x_block]*z_in_conflict_focus*(z_supports_y);

                        }
                        CYz[j] += CYz_reduction;
                    }
                }
            }
            cohesion_loop_time += omp_get_wtime() - time_start;
        }
    }

    // printf("nties = %f\n", contains_tie);

    printf("=========================================\n");
    printf("Allz OpenMP noties + nobeta Loop Times\n");
    printf("=========================================\n");

    printf("memops loop time: %.5fs\n", memops_loop_time);
    printf("conflict focus size loop time: %.5fs\n", conflict_loop_time);
    printf("cohesion matrix update loop time: %.5fs\n\n", cohesion_loop_time);

    // free up cache blocks
    _mm_free(distance_block);
    _mm_free(conflict_block);
}

/*
params
D    in  distance matrix: D(x,y) is distance between x and y (symmetric,
         but assumed to be stored in full)
beta in  conflict focus parameter: z is in focus of (x,y) if
         min(d(z,x),d(z,y)) <= beta * d(x,y)
n    in  number of points
C    out cohesion matrix: C(x,z) is z's support for x

Optimizations:
OpenMP, no-ties, nobeta, vectorized loop branching, Blocking, Masking, Auto-vectorization with  64-byte aligned arrays.
*/

void pald_allz_openmp_noties_nobeta_vecbranching(float* restrict D, float beta, int n, float* restrict C, int block_size, int nthreads) {
    /* TODO: Additional optimization strategies:
    *       1. Handle diagonal blocks separately so that inner loop can be fully vectorized for off-diagonal blocks.
    *       2. Add L1 blocking.
    *       3. Handle non-powers of two input dimensions using another method/set of nested loop to handle the ``remainder''.
    *       4. store conflict_block buffer as an int* type for faster increments -- but this requires one of two things: 1) a second float* buffer to hold 1./conflict_block or 2) on-the-fly floating-point casting and division. Latter is probably very slow.
    *
    */

    // declare indices
    unsigned int x, y, z, i, j, k, xb, yb, ib;
    // pre-allocate buffers for conflict focus and distance blocks
    float * conflict_block = (float *) _mm_malloc(block_size * block_size * sizeof(float),VECALIGN);
    float * distance_block = (float *) _mm_malloc(block_size * block_size * sizeof(float),VECALIGN);


    // initialize pointers for cache-block subcolumn vectors
    unsigned int y_block, x_block;
    double time_start = 0.0;
    double memops_loop_time = 0.0, conflict_loop_time = 0.0, cohesion_loop_time = 0.0;
    //up_left main block
    for (y = 0; y < n; y += block_size) {
        // loop over blocks of points X = (x,...,x+b-1)
        y_block = (block_size < n - y ? block_size : n - y);
        for (x = 0; x <= y; x += block_size) {
            time_start = omp_get_wtime();
            x_block = (block_size < n - x ? block_size : n - x);
            for (j = 0; j < y_block; ++j) {
                ib = (x == y ? j : x_block); // handle diagonal blocks
                // distance_block(:,j) = D(x:x+xb,y+j) in off-diagonal case
                memcpy(distance_block + j * x_block, D + x + (y + j) * n, ib * sizeof(float));
            }

            // compute block's conflict focus sizes by looping over all points z
            memset(conflict_block, 0, block_size * block_size * sizeof(float)); // clear old values

            memops_loop_time += omp_get_wtime() - time_start;
            time_start = omp_get_wtime();
            #pragma omp parallel for num_threads(nthreads) private(i, j, ib, z) reduction(+:conflict_block[:block_size*block_size]) schedule(static)
            for (z = 0; z < n; ++z) {
                float* DXz = D + x + z*n;
                float* DYz = D + y + z*n;
                for (j = 0; j < y_block; ++j) {
                    if(x == y){
                        for (i = 0; i < j; ++i) {
                            if (DYz[j] <= distance_block[i + j * x_block] || DXz[i] <= distance_block[i + j * x_block]){
                                ++conflict_block[i + j * x_block];
                            }

                        }
                    }
                    else{
                        for (i = 0; i < x_block; ++i) {
                            if (DYz[j] <= distance_block[i + j * x_block] || DXz[i] <= distance_block[i + j * x_block]){
                                ++conflict_block[i + j * x_block];
                            }

                        }
                    }
                }
            }
            #pragma omp parallel for num_threads(nthreads)
            for (k = 0; k < block_size*block_size; ++k)
                conflict_block[k] = 1.0f/conflict_block[k];
            // print_matrix(n, n, conflict_block);
            conflict_loop_time += omp_get_wtime() - time_start;
            time_start = omp_get_wtime();
            #pragma omp parallel num_threads(nthreads) private(i, j, ib, z)
            {
                float *CXz, *CYz, *DXz, *DYz;
                float z_supports_x;
                float z_supports_y;
                float z_supports_x_and_y;
                float z_in_conflict_focus;
                unsigned int mask_z_in_y_cutoff;
                unsigned int mask_z_in_x_cutoff;

                float CYz_reduction = 0.f;

                #pragma omp for nowait schedule(static)
                for (z = 0; z < n; ++z) {
                    // loop over all (i,j) pairs in block
                    DXz = D + x + z*n;
                    DYz = D + y + z*n;
                    CXz = C + x + z*n;
                    CYz = C + y + z*n;
                    for (j = 0; j < y_block; ++j) {
                        CYz_reduction = 0.f;
                        if(x == y){
                            for (i = 0; i < j; ++i) {
                                z_supports_x = DXz[i] < DYz[j];
                                z_supports_y = DXz[i] > DYz[j];

                                mask_z_in_x_cutoff = (DXz[i] <=  distance_block[i + j * x_block]);
                                mask_z_in_y_cutoff = (DYz[j] <=  distance_block[i + j * x_block]);
                                z_in_conflict_focus = mask_z_in_y_cutoff | mask_z_in_x_cutoff;

                                CXz[i] +=  conflict_block[i + j * x_block]*z_in_conflict_focus*(z_supports_x);
                                CYz_reduction +=  conflict_block[i + j * x_block]*z_in_conflict_focus*(z_supports_y);

                            }
                        }
                        else{
                            for (i = 0; i < x_block; ++i) {
                                z_supports_x = DXz[i] < DYz[j];
                                z_supports_y = DXz[i] > DYz[j];

                                mask_z_in_x_cutoff = (DXz[i] <=  distance_block[i + j * x_block]);
                                mask_z_in_y_cutoff = (DYz[j] <=  distance_block[i + j * x_block]);
                                z_in_conflict_focus = mask_z_in_y_cutoff | mask_z_in_x_cutoff;

                                CXz[i] +=  conflict_block[i + j * x_block]*z_in_conflict_focus*(z_supports_x);
                                CYz_reduction +=  conflict_block[i + j * x_block]*z_in_conflict_focus*(z_supports_y);

                            }
                        }
                        CYz[j] += CYz_reduction;
                    }
                }
            }
            cohesion_loop_time += omp_get_wtime() - time_start;
        }
    }

    // printf("nties = %f\n", contains_tie);

    printf("===========================================================\n");
    printf("Allz OpenMP noties + nobeta + vecbranching Loop Times\n");
    printf("===========================================================\n");

    printf("memops loop time: %.5fs\n", memops_loop_time);
    printf("conflict focus size loop time: %.5fs\n", conflict_loop_time);
    printf("cohesion matrix update loop time: %.5fs\n\n", cohesion_loop_time);

    // free up cache blocks
    _mm_free(distance_block);
    _mm_free(conflict_block);
}


/*
params
D    in  distance matrix: D(x,y) is distance between x and y (symmetric,
         but assumed to be stored in full)
beta in  conflict focus parameter: z is in focus of (x,y) if
         min(d(z,x),d(z,y)) <= beta * d(x,y)
n    in  number of points
C    out cohesion matrix: C(x,z) is z's support for x

Optimizations:
OpenMP, Blocking, Masking, Auto-vectorization with  64-byte aligned arrays.
*/

void pald_allz_openmp_experimental(float* restrict D, float beta, int n, float* restrict C, int block_size, int nthreads) {
    /* TODO: Additional optimization strategies:
    *       1. Handle diagonal blocks separately so that inner loop can be fully vectorized for off-diagonal blocks.
    *       2. Add L1 blocking.
    *       3. Handle non-powers of two input dimensions using another method/set of nested loop to handle the ``remainder''.
    *       4. store conflict_block buffer as an int* type for faster increments -- but this requires one of two things: 1) a second float* buffer to hold 1./conflict_block or 2) on-the-fly floating-point casting and division. Latter is probably very slow.
    *
    */

    // declare indices
    unsigned int x, y, z, i, j, k, xb, yb, ib;
    // pre-allocate buffers for conflict focus and distance blocks
    float * conflict_block = (float *) _mm_malloc(block_size * block_size * sizeof(float),VECALIGN);
    float * distance_block = (float *) _mm_malloc(block_size * block_size * sizeof(float),VECALIGN);


    // initialize pointers for cache-block subcolumn vectors
    unsigned int y_block, x_block;
    double time_start = 0.0;
    double memops_loop_time = 0.0, conflict_loop_time = 0.0, cohesion_loop_time = 0.0;
    //up_left main block
    for (y = 0; y < n; y += block_size) {
        // loop over blocks of points X = (x,...,x+b-1)
        y_block = (block_size < n - y ? block_size : n - y);
        for (x = 0; x <= y; x += block_size) {
            time_start = omp_get_wtime();
            x_block = (block_size < n - x ? block_size : n - x);
            for (j = 0; j < y_block; ++j) {
                ib = (x == y ? j : x_block); // handle diagonal blocks
                // distance_block(:,j) = D(x:x+xb,y+j) in off-diagonal case
                memcpy(distance_block + j * x_block, D + x + (y + j) * n, ib * sizeof(float));
            }

            // compute block's conflict focus sizes by looping over all points z
            memset(conflict_block, 0, block_size * block_size * sizeof(float)); // clear old values

            memops_loop_time += omp_get_wtime() - time_start;
            time_start = omp_get_wtime();
            #pragma omp parallel for num_threads(nthreads) private(i, j, ib, z) reduction(+:conflict_block[:block_size*block_size]) schedule(monotonic:dynamic,3)
            for (z = 0; z < n; ++z) {
                float* DXz = D + x + z*n;
                float* DYz = D + y + z*n;
                for (j = 0; j < y_block; ++j) {
                    ib = (x == y ? j : x_block);
                    for (i = 0; i < ib; ++i) {
                        if (DYz[j] <= beta * distance_block[i + j * x_block] || DXz[i] <= beta * distance_block[i + j * x_block]){
                            ++conflict_block[i + j * x_block];
                        }

                    }
                }
            }
            #pragma omp parallel for num_threads(nthreads)
            for (k = 0; k < block_size*block_size; ++k)
                conflict_block[k] = 1.0f/conflict_block[k];
            // print_matrix(n, n, conflict_block);
            conflict_loop_time += omp_get_wtime() - time_start;
            time_start = omp_get_wtime();
            #pragma omp parallel num_threads(nthreads) private(i, j, ib, z)
            {
                float *CXz, *CYz, *DXz, *DYz;
                float z_supports_x;
                float z_supports_y;
                float z_supports_x_and_y;
                float z_in_conflict_focus;
                unsigned int mask_z_in_y_cutoff;
                unsigned int mask_z_in_x_cutoff;

                float CYz_reduction = 0.f;
                float cutoff_distance = 0.f;

                #pragma omp for nowait schedule(monotonic:dynamic, 1)
                for (z = 0; z < n; ++z) {
                    // loop over all (i,j) pairs in block
                    DXz = D + x + z*n;
                    DYz = D + y + z*n;
                    CXz = C + x + z*n;
                    CYz = C + y + z*n;
                    for (j = 0; j < y_block; ++j) {
                        ib = (x == y ? j : x_block);
                        CYz_reduction = 0.f;
                        for (i = 0; i < ib; ++i) {
                            z_supports_x = DXz[i] < DYz[j];
                            z_supports_y = DXz[i] > DYz[j];
                            z_supports_x_and_y = (DXz[i] == DYz[j]) ? 0.5f : 0.0f;

                            cutoff_distance = beta*distance_block[i + j * x_block];
                            mask_z_in_x_cutoff = (DXz[i] <= cutoff_distance);
                            mask_z_in_y_cutoff = (DYz[j] <= cutoff_distance);
                            z_in_conflict_focus = mask_z_in_y_cutoff | mask_z_in_x_cutoff;

                            CXz[i] +=  conflict_block[i + j * x_block]*z_in_conflict_focus*(z_supports_x + z_supports_x_and_y);
                            CYz_reduction +=  conflict_block[i + j * x_block]*z_in_conflict_focus*(z_supports_y + z_supports_x_and_y);

                        }
                        CYz[j] += CYz_reduction;
                    }
                }
            }
            cohesion_loop_time += omp_get_wtime() - time_start;
        }
    }

    // printf("nties = %f\n", contains_tie);

    printf("=========================================\n");
    printf("Allz OpenMP Loop Times\n");
    printf("=========================================\n");

    printf("memops loop time: %.5fs\n", memops_loop_time);
    printf("conflict focus size loop time: %.5fs\n", conflict_loop_time);
    printf("cohesion matrix update loop time: %.5fs\n\n", cohesion_loop_time);

    // free up cache blocks
    _mm_free(distance_block);
    _mm_free(conflict_block);
}

void pald_allz_openmp(float* restrict D, float beta, int n, float* restrict C, int block_size, int nthreads) {
    // declare indices

    int x, y, z, i, j, k, xb, yb, ib;
    // pre-allocate buffers for conflict focus and distance blocks
    float *conflict_block = (float *) _mm_malloc(block_size * block_size * sizeof(float),VECALIGN);
    float *distance_block = (float *) _mm_malloc(block_size * block_size * sizeof(float),VECALIGN);
    double time_start = 0.0;
    double memops_loop_time = 0.0, conflict_loop_time = 0.0, cohesion_loop_time = 0.0;
    float dist_cutoff = 0., dist_cutoff_tmp = 0.;
    int y_block, x_block;
    //up_left main block

    for (y = 0; y < n; y += block_size) {
        // loop over blocks of points X = (x,...,x+b-1)
        y_block = (block_size < n - y ? block_size : n - y);
        for (x = 0; x <= y; x += block_size) {
            x_block = (block_size < n - x ? block_size : n - x);
            time_start = omp_get_wtime();
            for (j = 0; j < y_block; ++j) {
                ib = (x == y ? j : x_block); // handle diagonal blocks
                // distance_block(:,j) = D(x:x+xb,y+j) in off-diagonal case
                memcpy(distance_block + j * x_block, D + x + (y + j) * n, ib * sizeof(float));
            }
            // compute block's conflict focus sizes by looping over all points z
            // #pragma omp parallel for num_threads(nthreads) private(k) schedule(static)
            // for(k = 0; k < block_size*block_size; ++k)
            //     conflict_block[k] = 0;
            memset(conflict_block, 0, block_size * block_size * sizeof(float)); // clear old values
            // #pragma omp parallel for num_threads(nthreads) private(k)
            // for(k = 0; k < block_size*block_size; ++k)
            //     conflict_block[k] = 0.;
            memops_loop_time += omp_get_wtime() - time_start;

            time_start = omp_get_wtime();
            __assume(x_block % 16 == 0);
            __assume(n % 16 == 0);
            #pragma omp parallel for num_threads(nthreads) private(i, j, ib, z) reduction(+:conflict_block[:block_size*block_size]) schedule(monotonic:dynamic,3)
            for (z = 0; z < n; ++z) {

                float* DXz = D + x + z*n;
                float* DYz = D + y + z*n;
                // loop over all (i,j) pairs in block
                for (j = 0; j < y_block; ++j) {
                    ib = (x == y ? j : x_block);
                    for (i = 0; i < ib; ++i) {
                        if (DYz[j] <= beta * distance_block[i + j * x_block] || DXz[i] <= beta * distance_block[i + j * x_block]){
                            ++conflict_block[i + j * x_block];
                        }
                    }
                }
            }
            conflict_loop_time += omp_get_wtime() - time_start;

            #pragma omp parallel for num_threads(nthreads)
            for (k=0;k<block_size*block_size; ++k)
                conflict_block[k] = 1.0f/conflict_block[k];

            time_start = omp_get_wtime();
            #pragma omp parallel num_threads(nthreads) private(i, j, ib, z)
            {
                float *CXz, *CYz, *DXz, *DYz;
                // float *mask_z_in_conflict_focus = (float *) _mm_malloc(block_size * sizeof(float),VECALIGN);
                // float *mask_z_supports_x = (float *) _mm_malloc(block_size  * sizeof(float),VECALIGN);
                // float *mask_z_supports_x_and_y = (float *) _mm_malloc(block_size  * sizeof(float),VECALIGN);
                float z_in_conflict_focus;
                float z_supports_x;
                float z_supports_y;
                char mask_z_in_y_cutoff  = 0;
                char mask_z_in_x_cutoff = 0;

                float CYz_reduction = 0.f;
                float cutoff_distance = 0.f;
                float contains_tie = 0.f;

                #pragma omp for nowait schedule(monotonic:dynamic, 1)
                for (z = 0; z < n; ++z) {

                // loop over all (i,j) pairs in block
                    DXz = D + x + z*n;
                    DYz = D + y + z*n;

                    for (j = 0; j < y_block; ++j) {
                        ib = (x == y ? j : x_block);
                        // z supports y+j
                        CXz = C + x + z*n;
                        CYz = C + y + z*n;
                        // __assume(x_block % 16 == 0);
                        // __assume(n % 16 == 0);
                        CYz_reduction = 0.f;

                        for (i = 0; i < ib; ++i) {
                            // mask_z_supports_x is when z support x.
                            // mask_z_supports_x[i]= DXz[i] < DYz[j];
                            z_supports_x = DXz[i] < DYz[j];
                            z_supports_y = DXz[i] > DYz[j];
                            cutoff_distance = beta * distance_block[i + j * x_block];
                            mask_z_in_x_cutoff = (DXz[i] <= cutoff_distance);
                            mask_z_in_y_cutoff = (DYz[j] <= cutoff_distance);
                            z_in_conflict_focus = (mask_z_in_y_cutoff | mask_z_in_x_cutoff);
                            CXz[i] += conflict_block[i + j * x_block]*z_in_conflict_focus*(z_supports_x);
                            // 1 - mask_z_supports_x ==> z supports y and z supports both.
                            CYz_reduction +=  conflict_block[i + j * x_block]*z_in_conflict_focus*(z_supports_y);

                        }
                        CYz[j] += CYz_reduction;
                        // CXz = C + x + z*n;
                        // CYz = C + y + z*n;
                        // // __assume(x_block % 16 == 0);
                        // // __assume(n % 16 == 0);
                        // CYz_reduction = 0.f;
                        // for (i = 0; i<ib; ++i){
                        //     CXz[i] += conflict_block[i + j * x_block]*mask_z_in_conflict_focus[i]*(mask_z_supports_x[i]);
                        //     // 1 - mask_z_supports_x ==> z supports y and z supports both.
                        //     CYz_reduction +=  conflict_block[i + j * x_block]*mask_z_in_conflict_focus[i]*(1 - mask_z_supports_x[i]);
                        // }
                        // CYz[j] += CYz_reduction;

                        // for (i = 0; i < ib; ++i) {
                        //     // mask_z_supports_x_and_y is when z supports both x and y. Support should be divided between x and y.
                        //     mask_z_supports_x_and_y[i] = DXz[i] == DYz[j] ? 1.0f:0.0f;
                        //     contains_tie += mask_z_supports_x_and_y[i];
                        // }
                        // __assume(x_block % 16 == 0);
                        // __assume(n % 16 == 0);
                        // for (i = 0; i < ib; ++i){
                        //     cutoff_distance = beta * distance_block[i + j * x_block];
                        //     mask_z_in_x_cutoff = (DXz[i] <= cutoff_distance);
                        //     mask_z_in_y_cutoff = (DYz[j] <= cutoff_distance);
                        //     mask_z_in_conflict_focus[i] = (mask_z_in_y_cutoff | mask_z_in_x_cutoff);
                        // }
                        // CXz = C + x + z*n;
                        // CYz = C + y + z*n;
                        // // __assume(x_block % 16 == 0);
                        // // __assume(n % 16 == 0);
                        // CYz_reduction = 0.f;
                        // for (i = 0; i<ib; ++i){
                        //     CXz[i] += conflict_block[i + j * x_block]*mask_z_in_conflict_focus[i]*(mask_z_supports_x[i]);
                        //     // 1 - mask_z_supports_x ==> z supports y and z supports both.
                        //     CYz_reduction +=  conflict_block[i + j * x_block]*mask_z_in_conflict_focus[i]*(1 - mask_z_supports_x[i]);
                        // }
                        // CYz[j] += CYz_reduction;

                        // if (contains_tie > 0.5f){
                        //     contains_tie = 0.f;
                        //     CYz_reduction = CYz[j];
                        //     for (i = 0; i < ib; ++i){
                        //         CXz[i] += 0.5f * conflict_block[i + j * x_block]*mask_z_in_conflict_focus[i]*mask_z_supports_x_and_y[i];
                        //         // mask_z_supports_x contains ties so subtract 1/2.
                        //         CYz_reduction -= 0.5f * conflict_block[i + j * x_block]*mask_z_in_conflict_focus[i]*mask_z_supports_x_and_y[i];
                        //     }
                        //     CYz[j] = CYz_reduction;
                        // }
                    }
                }
                // _mm_free(mask_z_in_conflict_focus);
                // _mm_free(mask_z_supports_x);
                // _mm_free(mask_z_supports_x_and_y);
            }
            cohesion_loop_time += omp_get_wtime() - time_start;
        }
    }

    printf("==============================\n");
    printf("OMP Loop Times, nthreads: %3d\n", nthreads);
    printf("==============================\n");

    printf("memops loop time: %.5fs\n", memops_loop_time);
    printf("conflict focus size loop time: %.5fs\n", conflict_loop_time);
    printf("cohesion matrix update loop time: %.5fs\n\n", cohesion_loop_time);

    // free up cache blocks
    _mm_free(distance_block);
    _mm_free(conflict_block);
}

/*
params
D    in  distance matrix: D(x,y) is distance between x and y (symmetric,
         but assumed to be stored in full)
beta in  conflict focus parameter: z is in focus of (x,y) if
         min(d(z,x),d(z,y)) <= beta * d(x,y)
n    in  number of points
C    out cohesion matrix: C(x,z) is z's support for x
b    in  blocking parameter for cache efficiency
t    in  number of OMP threads to use
*/
void pald_allz_naive_openmp(float *D, float beta, int n, float *C, const int b, int t) {

    // pre-allocate conflict focus and distance cache blocks
    int *UXY = (int *) malloc(b * b * sizeof(int));
    float *DXY = (float *) malloc(b * b * sizeof(float));

    // loop over blocks of points Y = (y,...,y+b-1)
    for (int y = 0; y < n; y += b) {
        // define actual block size (for corner cases)
        int yb = (b < n - y ? b : n - y);

        // loop over blocks of points X = (x,...,x+b-1)
        for (int x = 0; x <= y; x += b) {
            // define actual block size (for corner cases)
            int xb = (b < n - x ? b : n - x);

            // copy distances into cache block one column at a time
            for (int j = 0; j < yb; j++) {
                // DXY(:,j) = D(x:x+xb,y+j) in off-diagonal case
                int ib = (x == y ? j : xb); // handle diagonal blocks
                memcpy(DXY + j * xb, D + x + (y + j) * n, ib * sizeof(float));
            }

            // DEBUG: print out DXY cache block
            /*printf("x %d y %d xb %d yb %d\n",x,y,xb,yb);
            printf("\nDXY\n");
            for (int i = 0; i < xb; i++)
            {
                for (int j = 0; j < yb; j++)
                {
                    printf("%f ", DXY[i+j*xb]);
                }
                printf("\n");
            }
            printf("\n");*/

            // compute block's conflict focus sizes by looping over all points z
            memset(UXY, 0, b * b * sizeof(int)); // clear old values
            #pragma omp parallel for num_threads(t) reduction(+:UXY[:b*b])
            for (int z = 0; z < n; z++) {
                // set pointers to subcolumns of D
                float* DXz = D + x + z*n;
                float* DYz = D + y + z*n;
                // loop over all (i,j) pairs in block
                for (int j = 0; j < yb; j++) {
                    int ib = (x == y ? j : xb); // handle diagonal blocks
                    for (int i = 0; i < ib; i++)
                        // DXY[i+j*xb] is distance between x+i and y+j
                        // DXz[i] is distance between x+i and z
                        // DYz[j] is distance between y+j and z

                        // determine if z is in conflict focus of x+i and y+j
                        if (DYz[j] <= beta * DXY[i + j * xb] || DXz[i] <= beta * DXY[i + j * xb])
                            UXY[i + j * xb]++;
                }
            }

            // DEBUG: print out UXY cache block
            /*for (int i = 0; i < xb; i++)
            {
                for (int j = 0; j < yb; j++)
                {
                    printf("%d ", UXY[i+j*xb]);
                }
                printf("\n");
            }
            printf("\n");*/

            // update cohesion values according to conflicts between X and Y
            // by looping over all points z
            #pragma omp parallel for num_threads(t)
            for (int z = 0; z < n; z++) {
                // set pointers to subcolumns of D
                float* DXz = D + x + z*n;
                float* DYz = D + y + z*n;
                // set pointers to subcolumns of C
                float* CXz = C + x + z*n;
                float* CYz = C + y + z*n;
                // loop over all (i,j) pairs in block
                for (int j = 0; j < yb; j++) {
                    int ib = (x == y ? j : xb); // handle diagonal blocks
                    for (int i = 0; i < ib; i++) {
                        // DXY[i+j*xb] is distance between x+i and y+j
                        // DXz[i] is distance between x+i and z
                        // DYz[j] is distance between y+j and z

                        // check if z is in conflict of (x+i,y+j)
                        if (DYz[j] <= beta * DXY[i + j * xb] || DXz[i] <= beta * DXY[i + j * xb]) {
                            // z supports x+i
                            if (DXz[i] < DYz[j])
                                CXz[i] += 1.0f / UXY[i + j * xb];
                                // z supports y+j
                            else if (DYz[j] < DXz[i])
                                CYz[j] += 1.0f / UXY[i + j * xb];
                                // z splits its support
                            else {
                                CXz[i] += 0.5f / UXY[i + j * xb];
                                CYz[j] += 0.5f / UXY[i + j * xb];
                            }
                        }
                    }
                }
            }
        }
    }

    // free up cache blocks
    free(DXY);
    free(UXY);
    // print out timing results before returning
}

void pald_triplet_naive(float *D, float beta, int n, float *C){
    //TODO: Naive sequential triplet code.
    float* conflict_matrix = (float *)  malloc(n * n * sizeof(float));
    memset(conflict_matrix, 0, n * n * sizeof(float));

    double conflict_loop_time, cohesion_loop_time;
    double time_start = omp_get_wtime();
    for (int i = 0; i < n; ++i){
        for (int j = i + 1; j < n; ++j){
            conflict_matrix[i + j * n] = 2.;
        }
        // for (int j = i + 1; j < n; ++j){
        //     conflict_matrix[i + j * n] = 2.;
        // }
    }
    // print_matrix(n,conflict_matrix);
    // Compute conflict focus size.

    for(int x = 0; x < n - 1; ++x){
        for(int y = x + 1; y < n; ++y){
            for(int z = y + 1; z < n; ++z){
                if (D[x + y * n] < D[x + z * n] && D[x + y * n] < D[y + z * n]) { // x and y are close
                    conflict_matrix[x + z * n] += 1;
                    conflict_matrix[y + z * n] += 1;
                }
                else if(D[x + z * n] < D[x + y * n] && D[x + z * n] < D[y + z * n]){ // x and z are close
                    conflict_matrix[x + y * n] += 1;
                    conflict_matrix[y + z * n] += 1;
                }
                else if (D[y + z * n] < D[x + y * n] && D[y + z * n] < D[x + z * n]){ // y and z are close
                    conflict_matrix[x + y * n] += 1;
                    conflict_matrix[x + z * n] += 1;
                }
                else{ // we have pairwise/triplet ties.
                    conflict_matrix[x + y * n] += 1;
                    conflict_matrix[x + z * n] += 1;
                    conflict_matrix[y + z * n] += 1;
                }
            }

        }
    }
    conflict_loop_time = omp_get_wtime() - time_start;
    time_start = omp_get_wtime();
    // print_matrix(n, n, conflict_matrix);
    // initialize diagonal of C.
    float sum;
    for (int i = 0; i < n; ++i){
        sum = 0.f;
        for (int j = 0; j < i; ++j){
            sum += 1.f / conflict_matrix[j + i * n];
        }

        for (int j = i + 1; j < n; ++j){
            sum += 1.f / conflict_matrix[i + j * n];
        }
        C[i + i * n] = sum;
    }

    // Compute cohesion matrix.
    for(int x = 0; x < n - 1; ++x){
        for(int y = x + 1; y < n; ++y){
            for(int z = y + 1; z < n; ++z){
                if (D[x + y * n] < D[x + z * n] && D[x + y * n] < D[y + z * n]) { // x and y are close
                    C[x + y * n] += 1.f/(conflict_matrix[x + z * n]);
                    C[y + x * n] += 1.f/(conflict_matrix[y + z * n]);
                }
                else if(D[x + z * n] < D[x + y * n] && D[x + z * n] < D[y + z *n]){ // x and z are close
                    C[x + z * n] += 1.f/(conflict_matrix[x + y * n]);
                    C[z + x * n] += 1.f/(conflict_matrix[y + z * n]);
                }
                else if (D[y + z * n] < D[x + y * n] && D[y + z * n] < D[x + z * n]){ // y and z are close
                    C[y + z * n] += 1.f/(conflict_matrix[x + y * n]);
                    C[z + y * n] += 1.f/(conflict_matrix[x + z * n]);
                }
                else {
                    if (D[x + y * n] == D[x + z * n]){
                        C[x + y * n] += 1.f/conflict_matrix[x + z * n];
                        C[y + x * n] += .5f/conflict_matrix[y + z * n];

                        C[x + z * n] += 1.f/conflict_matrix[x + y * n];
                        C[z + x * n] += .5f/conflict_matrix[y + z * n];
                    }
                    if (D[x + z * n] == D[y + z * n]){ //dist xz == dist yz
                        C[x + z * n] += .5f/(conflict_matrix[x + y * n]);
                        C[z + x * n] += 1.f/(conflict_matrix[y + z * n]);

                        C[y + z * n] += .5f/conflict_matrix[x + y * n];
                        C[z + y * n] += 1.f/conflict_matrix[x + z * n];
                    }
                    if (D[x + y * n] == D[y + z * n]){
                        C[x + y * n] += .5f/conflict_matrix[x + z * n];
                        C[y + x * n] += 1.f/conflict_matrix[y + z * n];

                        C[y + z * n] += 1.f/conflict_matrix[x + y * n];
                        C[z + y * n] += .5f/conflict_matrix[x + z * n];
                    }
                }
            }
        }
    }
    cohesion_loop_time = omp_get_wtime() - time_start;
    // print_matrix(n, C);

    printf("======================================\n");
    printf("Seq. Naive Triplet Loop Times\n");
    printf("======================================\n");

    // printf("memops loop time: %.5fs\n", memops_loop_time);
    printf("conflict focus size loop time: %.5fs\n", conflict_loop_time);
    printf("cohesion matrix update loop time: %.5fs\n\n", cohesion_loop_time);
}

void pald_triplet_naive_openmp(float *D, float beta, int n, float *C, int nthreads){
    //TODO: Naive OpenMP triplet code.
        //TODO: Naive sequential triplet code.
    float* conflict_matrix = (float *)  malloc(n * n * sizeof(float));
    memset(conflict_matrix, 0, n * n * sizeof(float));
    long int len = n * n;
    double conflict_loop_time, cohesion_loop_time;
    double time_start = omp_get_wtime();
    for (int i = 0; i < n; ++i){
        for (int j = i + 1; j < n; ++j){
            conflict_matrix[i + j * n] = 2.;
        }
        // for (int j = i + 1; j < n; ++j){
        //     conflict_matrix[i + j * n] = 2.;
        // }
    }
    // print_matrix(n,conflict_matrix);
    // Compute conflict focus size.
    // printf("conflict_matrix start.\n");
    #pragma omp parallel for num_threads(nthreads) reduction(+:conflict_matrix[:len])
    for(int x = 0; x < n - 1; ++x){
        for(int y = x + 1; y < n; ++y){
            for(int z = y + 1; z < n; ++z){
                if (D[x + y * n] < D[x + z * n] && D[x + y * n] < D[y + z * n]) { // x and y are close
                    conflict_matrix[x + z * n] += 1;
                    conflict_matrix[y + z * n] += 1;
                }
                else if(D[x + z * n] < D[x + y * n] && D[x + z * n] < D[y + z * n]){ // x and z are close
                    conflict_matrix[x + y * n] += 1;
                    conflict_matrix[y + z * n] += 1;
                }
                else{ // y and z are close
                    conflict_matrix[x + y * n] += 1;
                    conflict_matrix[x + z * n] += 1;
                }
            }

        }
    }
    // printf("conflict_matrix done.\n");
    conflict_loop_time = omp_get_wtime() - time_start;
    time_start = omp_get_wtime();
    // print_matrix(n,conflict_matrix);
    // initialize diagonal of C.
    // float sum;
    // for (int i = 0; i < n; ++i){
    //     sum = 0.f;
    //     for (int j = 0; j < i; ++j){
    //         C[i + i * n] += 1.f / conflict_matrix[i + j * n];
    //     }

    //     for (int j = i + 1; j < n; ++j){
    //         C[i + i * n] += 1.f / conflict_matrix[i + j * n];
    //     }
    // }

    // Compute cohesion matrix.
    #pragma omp parallel for num_threads(nthreads) reduction(+:C[:len])
    for(int x = 0; x < n - 1; ++x){
        for(int y = x + 1; y < n; ++y){
            for(int z = y + 1; z < n; ++z){
                if (D[x + y * n] < D[x + z * n] && D[x + y * n] < D[y + z * n]) { // x and y are close
                    C[x + y * n] += 1.f/(conflict_matrix[x + z * n]);
                    C[y + x * n] += 1.f/(conflict_matrix[y + z * n]);
                    // C[x + y * n] += 1.f/(conflict_matrix[x + z * n] + 2.f);
                    // C[y + x * n] += 1.f/(conflict_matrix[y + z * n] + 2.f);
                }
                else if(D[x + z * n] < D[x + y * n] && D[x + z * n] < D[y + z *n]){ // x and z are close
                    C[x + z * n] += 1.f/(conflict_matrix[x + y * n]);
                    C[z + x * n] += 1.f/(conflict_matrix[y + z * n]);
                    // C[x + z * n] += 1.f/(conflict_matrix[x + y * n] + 2.f);
                    // C[z + x * n] += 1.f/(conflict_matrix[y + z * n] + 2.f);
                }
                else{ // y and z are close
                    C[y + z * n] += 1.f/(conflict_matrix[x + y * n]);
                    C[z + y * n] += 1.f/(conflict_matrix[x + z * n]);
                    // C[y + z * n] += 1.f/(conflict_matrix[x + y * n] + 2.f);
                    // C[z + y * n] += 1.f/(conflict_matrix[x + z * n] + 2.f);
                }
            }
        }
    }
    cohesion_loop_time = omp_get_wtime() - time_start;
    // print_matrix(n, C);

    printf("====================================================\n");
    printf("Naive Triplet OMP Loop Times threads: %d\n", nthreads);
    printf("====================================================\n");

    // printf("memops loop time: %.5fs\n", memops_loop_time);
    printf("conflict focus size loop time: %.5fs\n", conflict_loop_time);
    printf("cohesion matrix update loop time: %.5fs\n\n", cohesion_loop_time);
}

void pald_triplet_blocked_powersoftwo(float* restrict D, float beta, int n, float* restrict C, int block_size){
    //TODO: Optimized sequential triplet code.
    float* conflict_matrix = (float *)  _mm_malloc(n * n * sizeof(float), VECALIGN);
    memset(conflict_matrix, 0, n * n * sizeof(float));

    float* distance_xy_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
    float* distance_xz_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
    float* distance_yz_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);

    float *conflict_xy_block, *conflict_xz_block, *conflict_yz_block;
    float *cohesion_xy_block, *cohesion_yx_block, *cohesion_xz_block, *cohesion_zx_block, *cohesion_yz_block, *cohesion_zy_block;
    char print_out = 0;
    double time_start = 0.0;
    double memops_loop_time = 0.0, conflict_loop_time = 0.0, cohesion_loop_time = 0.0;
    time_start = omp_get_wtime();
    for (int i = 0; i < n; ++i){
        for (int j = i + 1; j < n; ++j){
            conflict_matrix[j + i * n] = 2.;
        }
    }
    conflict_loop_time += omp_get_wtime() - time_start;
    if(print_out)
        print_matrix(n, n, conflict_matrix);

    int xb, yb, zb, x, y, z;
    int i, j, k;
    int size_xy = block_size, size_xz = block_size, size_yz = block_size;
    int x_block, y_block, z_block;
    int xend, ystart, zstart;
    float xy_reduction, yx_reduction;
    // compute conflict focus sizes.
    int iters = 0;
    for(xb = 0; xb < n; xb += block_size){
        x_block = ((xb + block_size) < n) ? block_size : (n - xb);
        for(yb = xb; yb < n; yb += block_size){
            time_start = omp_get_wtime();
            y_block = ((yb + block_size) < n) ? block_size : (n - yb);
            for (i = 0; i < block_size; ++i){
                //size_xy = (xb == yb) ? i : block_size;
                memcpy(distance_xy_block + i * block_size, D + yb + (xb + i) * n, sizeof(float)*size_xy);
            }
            memops_loop_time += omp_get_wtime() - time_start;
            // for(i = 0; i < block_size*block_size; ++i){
            //     printf("%.7f\n",distance_xy_block[i]);
            // }
            // copy DXY block from D.
            for(zb = yb; zb < n; zb += block_size){
                z_block = ((zb + block_size) < n) ? block_size : (n - zb);
                // if(xb == yb && yb == zb){
                //     conflict_yz_block = conflict_xy_block;
                //     conflict_xz_block = conflict_xy_block;
                // }
                // else if(xb == yb){
                //     conflict_yz_block = conflict_xz_block;
                // }
                // else if(yb == zb){
                //     conflict_xy_block = conflict_xz_block;
                // }
                //copy DXZ and DYZ blocks from D.c
                time_start = omp_get_wtime();
                for (i = 0; i < block_size; ++i){
                    // size_xz = (xb == zb) ? i : block_size;
                    // size_yz = (yb == zb) ? i : block_size;
                    memcpy(distance_xz_block + i * block_size, D + zb + (xb + i) * n, sizeof(float)*size_xz);
                    memcpy(distance_yz_block + i * block_size, D + zb + (yb + i) * n, sizeof(float)*size_yz);
                }
                memops_loop_time += omp_get_wtime() - time_start;
                if(print_out){
                    print_matrix(block_size, block_size, distance_xy_block);
                    print_matrix(block_size, block_size, distance_xz_block);
                    print_matrix(block_size, block_size, distance_yz_block);
                    printf("(xb:%d, yb:%d, zb:%d)\n", xb, yb, zb);
                    print_matrix(block_size, n, conflict_matrix + yb + xb * n);
                    print_matrix(block_size, n, conflict_matrix + zb + xb * n);
                    print_matrix(block_size, n, conflict_matrix + zb + yb * n);
                    print_matrix(n, n, conflict_matrix);
                }
                time_start = omp_get_wtime();
                conflict_xy_block = conflict_matrix + yb + xb * n;
                conflict_xz_block = conflict_matrix + zb + xb * n;
                conflict_yz_block = conflict_matrix + zb + yb * n;
                // printf("(%d, %d, %d)\n", xb, yb, zb);
                xend = block_size;
                ystart = 0;
                zstart = 0;
                if(xb == yb && yb == zb){
                    xend = block_size - 1;
                }
                for(x = 0; x < xend; ++x){
                    if(xb == yb){
                        ystart = x + 1;
                        conflict_yz_block += ystart*n;
                    }
                    for(y = ystart; y < block_size; ++y){
                        xy_reduction = 0.f;
                        if(yb == zb){
                            zstart = y + 1;
                        }
                        for(z = zstart; z < block_size; ++z){
                            // update conflict matrix blocks.
                            if(print_out)
                                printf("(x:%d, y:%d, z:%d) : (%.2f, %.2f, %.2f)\n", x, y, z, distance_xy_block[y + x * block_size], distance_xz_block[z + x * block_size], distance_yz_block[z + y * block_size]);
                            if(distance_xy_block[y + x * block_size] < distance_xz_block[z + x * block_size] && distance_xy_block[y + x * block_size] < distance_yz_block[z + y * block_size]){
                                ++conflict_xz_block[z];
                                ++conflict_yz_block[z];
                            }
                            else if(distance_xz_block[z + x * block_size] < distance_xy_block[y + x * block_size] && distance_xz_block[z + x * block_size] < distance_yz_block[z + y * block_size]){
                                ++xy_reduction;
                                ++conflict_yz_block[z];
                            }
                            else if(distance_yz_block[z + y * block_size] < distance_xz_block[z + x * block_size] && distance_yz_block[z + y * block_size] < distance_xy_block[y + x * block_size]){
                                ++xy_reduction;
                                ++conflict_xz_block[z];
                            }
                            else{
                                ++xy_reduction;
                                ++conflict_xz_block[z];
                                ++conflict_yz_block[z];
                            }
                        }
                        // print_matrix(block_size, n, conflict_xy_block);
                        conflict_xy_block[y] += xy_reduction;
                        conflict_yz_block += n;
                    }
                    conflict_xz_block += n;
                    conflict_xy_block += n;
                    conflict_yz_block = conflict_matrix + zb + yb * n;
                }
                if(print_out){
                    print_matrix(n, n, conflict_matrix);
                    // printf("\n\n");
                    printf("iters:%d\n", iters);
                    // if(iters == 2)
                    //     exit(-1);
                }
                // if(iters == 7){
                //     exit(-1);
                // }
                iters++;
                conflict_loop_time += omp_get_wtime() - time_start;
            }
        }
    }
    // print_matrix(n, n, conflict_matrix);
    // printf("\n\n");
        // initialize diagonal of C.
    float sum;
    time_start = omp_get_wtime();
    for (i = 0; i < n; ++i){
        sum = 0.f;
        for (j = 0; j < i; ++j){
            sum += 1.f / conflict_matrix[i + j * n];
        }

        for (j = i + 1; j < n; ++j){
            sum += 1.f / conflict_matrix[j + i * n];
        }
        C[i + i * n] = sum;
    }
    conflict_loop_time += omp_get_wtime() - time_start;
    iters = 0;
    for(xb = 0; xb < n; xb += block_size){
        for(yb = xb; yb < n; yb += block_size){
            time_start = omp_get_wtime();
            for (i = 0; i < block_size; ++i){
                //size_xy = (xb == yb) ? i : block_size;
                memcpy(distance_xy_block + i * block_size, D + yb + (xb + i) * n, sizeof(float)*size_xy);
            }
            memops_loop_time += omp_get_wtime() - time_start;
            for(zb = yb; zb < n; zb += block_size){
                if(print_out){
                    print_matrix(block_size, n, cohesion_xy_block);
                    print_matrix(block_size, n, cohesion_yx_block);
                    print_matrix(block_size, n, cohesion_xz_block);
                    print_matrix(block_size, n, cohesion_zx_block);
                    print_matrix(block_size, n, cohesion_yz_block);
                    print_matrix(block_size, n, cohesion_zy_block);
                    printf("(xb:%d, yb:%d, zb:%d)\n", xb, yb, zb);
                }
                time_start = omp_get_wtime();
                for (i = 0; i < block_size; ++i){
                    // size_xz = (xb == zb) ? i : block_size;
                    // size_yz = (yb == zb) ? i : block_size;
                    memcpy(distance_xz_block + i * block_size, D + zb + (xb + i) * n, sizeof(float)*size_xz);
                    memcpy(distance_yz_block + i * block_size, D + zb + (yb + i) * n, sizeof(float)*size_yz);
                }
                memops_loop_time += omp_get_wtime() - time_start;
                time_start = omp_get_wtime();
                conflict_xy_block = conflict_matrix + yb + xb * n;
                conflict_xz_block = conflict_matrix + zb + xb * n;
                conflict_yz_block = conflict_matrix + zb + yb * n;

                cohesion_xy_block = C + yb + xb * n;
                cohesion_yx_block = C + xb + yb * n;
                cohesion_xz_block = C + zb + xb * n;
                cohesion_zx_block = C + xb + zb * n;
                cohesion_yz_block = C + zb + yb * n;
                cohesion_zy_block = C + yb + zb * n;
                xend = block_size;
                ystart = 0;
                zstart = 0;
                if(xb == yb && yb == zb){
                    xend = block_size - 1;
                }
                for(x = 0; x < xend; ++x){
                    if(xb == yb){
                        ystart = x + 1;
                        conflict_yz_block += ystart*n;
                        cohesion_yz_block += ystart*n;
                        cohesion_yx_block += ystart*n;
                    }
                    for(y = ystart; y < block_size; ++y){
                        xy_reduction = 0.f;
                        yx_reduction = 0.f;
                        if(yb == zb){
                            zstart = y + 1;
                            cohesion_zx_block += zstart*n;
                            cohesion_zy_block += zstart*n;
                        }
                        for(z = zstart; z < block_size; ++z){
                            // update cohesion matrix blocks.
                            if(print_out)
                                printf("(x:%d, y:%d, z:%d) : (%.2f, %.2f, %.2f)\n", x, y, z, distance_xy_block[y + x * block_size], distance_xz_block[z + x * block_size], distance_yz_block[z + y * block_size]);
                            if(distance_xy_block[y + x * block_size] < distance_xz_block[z + x * block_size] && distance_xy_block[y + x * block_size] < distance_yz_block[z + y * block_size]){
                                yx_reduction += 1.f/conflict_xz_block[z];
                                xy_reduction += 1.f/conflict_yz_block[z];
                            }
                            else if(distance_xz_block[z + x * block_size] < distance_xy_block[y + x * block_size] && distance_xz_block[z + x * block_size] < distance_yz_block[z + y * block_size]){
                                cohesion_zx_block[x] += 1.f/conflict_xy_block[y];
                                cohesion_xz_block[z] += 1.f/conflict_yz_block[z];
                            }
                            else if(distance_yz_block[z + y * block_size] < distance_xz_block[z + x * block_size] && distance_yz_block[z + y * block_size] < distance_xy_block[y + x * block_size]){
                                cohesion_zy_block[y] += 1.f/conflict_xy_block[y];
                                cohesion_yz_block[z] += 1.f/conflict_xz_block[z];
                            }
                            else{
                                if (distance_xy_block[y + x * block_size] == distance_xz_block[z + x * block_size]){
                                    yx_reduction += 1.f/conflict_xz_block[z];
                                    xy_reduction += .5f/conflict_yz_block[z];
                                    cohesion_xz_block[z] += .5f/conflict_yz_block[z];
                                    cohesion_zx_block[x] += 1.f/conflict_xy_block[y];
                                }

                                if (distance_xz_block[z + x * block_size] == distance_yz_block[z + y * block_size]){
                                    cohesion_zx_block[x] += .5f / conflict_xy_block[y];
                                    cohesion_zy_block[y] += .5f / conflict_xy_block[y];

                                    cohesion_xz_block[z] += 1.f / conflict_yz_block[z];
                                    cohesion_yz_block[z] += 1.f / conflict_xz_block[z];
                                }

                                if (distance_xy_block[y + x * block_size] == distance_yz_block[z + y * block_size]){
                                    yx_reduction += .5f/conflict_xz_block[z];
                                    xy_reduction += 1.f/conflict_yz_block[z];

                                    cohesion_zy_block[y] += 1.f/conflict_xy_block[y];
                                    cohesion_yz_block[z] += .5f/conflict_xz_block[z];
                                }
                            }
                            cohesion_zx_block += n;
                            cohesion_zy_block += n;
                        }
                        conflict_yz_block += n;

                        cohesion_xy_block[y] += xy_reduction;
                        cohesion_yx_block[x] += yx_reduction;
                        cohesion_yx_block += n;
                        cohesion_yz_block += n;
                        cohesion_zx_block = C + xb + zb * n;
                        cohesion_zy_block = C + yb + zb * n;
                    }
                    conflict_xz_block += n;
                    conflict_xy_block += n;
                    conflict_yz_block = conflict_matrix + zb + yb * n;

                    cohesion_xz_block += n;
                    cohesion_xy_block += n;
                    cohesion_yx_block = C + xb + yb * n;
                    cohesion_yz_block = C + zb + yb * n;
                }
                if(print_out)
                    print_matrix(n, n, C);
                // if(iters == 15)
                //     exit(-1);
                iters++;
                cohesion_loop_time += omp_get_wtime() - time_start;
            }
        }
    }


    printf("====================================================\n");
    printf("Seq. Triplet Blocked Powers of Two Loop Times\n");
    printf("====================================================\n");

    printf("memops loop time: %.5fs\n", memops_loop_time);
    printf("conflict focus size loop time: %.5fs\n", conflict_loop_time);
    printf("cohesion matrix update loop time: %.5fs\n\n", cohesion_loop_time);

    _mm_free(conflict_matrix);
}

void pald_triplet_blocked(float* restrict D, float beta, int n, float* restrict C, int block_size){
    //TODO: Optimized sequential triplet code.
    float* conflict_matrix = (float *)  _mm_malloc(n * n * sizeof(float), VECALIGN);
    memset(conflict_matrix, 0, n * n * sizeof(float));

    float* distance_xy_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
    float* distance_xz_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
    float* distance_yz_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);

    float *conflict_xy_block, *conflict_xz_block, *conflict_yz_block;
    float *cohesion_xy_block, *cohesion_yx_block, *cohesion_xz_block, *cohesion_zx_block, *cohesion_yz_block, *cohesion_zy_block;
    char print_out = 0;
    double time_start = 0.0;
    double memops_loop_time = 0.0, conflict_loop_time = 0.0, cohesion_loop_time = 0.0;
    time_start = omp_get_wtime();
    for (int i = 0; i < n; ++i){
        for (int j = i + 1; j < n; ++j){
            conflict_matrix[j + i * n] = 2.;
        }
    }
    conflict_loop_time += omp_get_wtime() - time_start;
    if(print_out)
        print_matrix(n, n, conflict_matrix);

    int xb, yb, zb, x, y, z;
    int i, j, k;
    int size_xy = block_size, size_xz = block_size, size_yz = block_size;
    int x_block, y_block, z_block;
    int xend, ystart, zstart;
    float xy_reduction, yx_reduction;
    // compute conflict focus sizes.
    int iters = 0;
    for(xb = 0; xb < n; xb += block_size){
        x_block = ((xb + block_size) < n) ? block_size : (n - xb);
        for(yb = xb; yb < n; yb += block_size){
            time_start = omp_get_wtime();
            y_block = ((yb + block_size) < n) ? block_size : (n - yb);
            for (i = 0; i < x_block; ++i){
                //size_xy = (xb == yb) ? i : block_size;
                memcpy(distance_xy_block + i * block_size, D + yb + (xb + i) * n, sizeof(float)*y_block);
            }
            memops_loop_time += omp_get_wtime() - time_start;
            // for(i = 0; i < block_size*block_size; ++i){
            //     printf("%.7f\n",distance_xy_block[i]);
            // }
            // copy DXY block from D.
            for(zb = yb; zb < n; zb += block_size){
                z_block = ((zb + block_size) < n) ? block_size : (n - zb);
                // if(xb == yb && yb == zb){
                //     conflict_yz_block = conflict_xy_block;
                //     conflict_xz_block = conflict_xy_block;
                // }
                // else if(xb == yb){
                //     conflict_yz_block = conflict_xz_block;
                // }
                // else if(yb == zb){
                //     conflict_xy_block = conflict_xz_block;
                // }
                //copy DXZ and DYZ blocks from D.c
                time_start = omp_get_wtime();
                for (i = 0; i < x_block; ++i){
                    // size_xz = (xb == zb) ? i : block_size;
                    // size_yz = (yb == zb) ? i : block_size;
                    memcpy(distance_xz_block + i * block_size, D + zb + (xb + i) * n, sizeof(float)*z_block);
                }
                for (i = 0; i < y_block; ++i){
                    memcpy(distance_yz_block + i * block_size, D + zb + (yb + i) * n, sizeof(float)*z_block);
                }
                memops_loop_time += omp_get_wtime() - time_start;
                if(print_out){
                    print_matrix(block_size, block_size, distance_xy_block);
                    print_matrix(block_size, block_size, distance_xz_block);
                    print_matrix(block_size, block_size, distance_yz_block);
                    printf("(xb:%d, yb:%d, zb:%d)\n", xb, yb, zb);
                    print_matrix(block_size, n, conflict_matrix + yb + xb * n);
                    print_matrix(block_size, n, conflict_matrix + zb + xb * n);
                    print_matrix(block_size, n, conflict_matrix + zb + yb * n);
                    print_matrix(n, n, conflict_matrix);
                }
                time_start = omp_get_wtime();
                conflict_xy_block = conflict_matrix + yb + xb * n;
                conflict_xz_block = conflict_matrix + zb + xb * n;
                conflict_yz_block = conflict_matrix + zb + yb * n;
                // printf("(%d, %d, %d)\n", xb, yb, zb);
                xend = x_block;
                ystart = 0;
                zstart = 0;
                if(xb == yb && yb == zb){
                    xend = x_block - 1;
                }
                for(x = 0; x < xend; ++x){
                    if(xb == yb){
                        ystart = x + 1;
                        conflict_yz_block += ystart*n;
                    }
                    for(y = ystart; y < y_block; ++y){
                        xy_reduction = 0.f;
                        if(yb == zb){
                            zstart = y + 1;
                        }
                        for(z = zstart; z < z_block; ++z){
                            // update conflict matrix blocks.
                            if(print_out)
                                printf("(x:%d, y:%d, z:%d) : (%.2f, %.2f, %.2f)\n", x, y, z, distance_xy_block[y + x * block_size], distance_xz_block[z + x * block_size], distance_yz_block[z + y * block_size]);
                            if(distance_xy_block[y + x * block_size] < distance_xz_block[z + x * block_size] && distance_xy_block[y + x * block_size] < distance_yz_block[z + y * block_size]){
                                ++conflict_xz_block[z];
                                ++conflict_yz_block[z];
                            }
                            else if(distance_xz_block[z + x * block_size] < distance_xy_block[y + x * block_size] && distance_xz_block[z + x * block_size] < distance_yz_block[z + y * block_size]){
                                ++xy_reduction;
                                ++conflict_yz_block[z];
                            }
                            else if(distance_yz_block[z + y * block_size] < distance_xz_block[z + x * block_size] && distance_yz_block[z + y * block_size] < distance_xy_block[y + x * block_size]){
                                ++xy_reduction;
                                ++conflict_xz_block[z];
                            }
                            else{
                                continue;
                                ++xy_reduction;
                                ++conflict_xz_block[z];
                                ++conflict_yz_block[z];
                            }
                        }
                        // print_matrix(block_size, n, conflict_xy_block);
                        conflict_xy_block[y] += xy_reduction;
                        conflict_yz_block += n;
                    }
                    conflict_xz_block += n;
                    conflict_xy_block += n;
                    conflict_yz_block = conflict_matrix + zb + yb * n;
                }
                if(print_out){
                    print_matrix(n, n, conflict_matrix);
                    // printf("\n\n");
                    printf("iters:%d\n", iters);
                    // if(iters == 2)
                    //     exit(-1);
                }
                // if(iters == 7){
                //     exit(-1);
                // }
                iters++;
                conflict_loop_time += omp_get_wtime() - time_start;
            }
        }
    }
    // print_matrix(n, n, conflict_matrix);
    // printf("\n\n");
    // return;
        // initialize diagonal of C.
    float sum;
    time_start = omp_get_wtime();
    for (i = 0; i < n; ++i){
        sum = 0.f;
        for (j = 0; j < i; ++j){
            sum += 1.f / conflict_matrix[i + j * n];
        }

        for (j = i + 1; j < n; ++j){
            sum += 1.f / conflict_matrix[j + i * n];
        }
        C[i + i * n] = sum;
    }
    conflict_loop_time += omp_get_wtime() - time_start;
    iters = 0;
    for(xb = 0; xb < n; xb += block_size){
        x_block = ((xb + block_size) < n) ? block_size : (n - xb);
        for(yb = xb; yb < n; yb += block_size){
            time_start = omp_get_wtime();
            y_block = ((yb + block_size) < n) ? block_size : (n - yb);

            for (i = 0; i < x_block; ++i){
                //size_xy = (xb == yb) ? i : block_size;
                memcpy(distance_xy_block + i * block_size, D + yb + (xb + i) * n, sizeof(float)*y_block);
            }
            memops_loop_time += omp_get_wtime() - time_start;
            for(zb = yb; zb < n; zb += block_size){
                z_block = ((zb + block_size) < n) ? block_size : (n - zb);
                if(print_out){
                    print_matrix(block_size, n, cohesion_xy_block);
                    print_matrix(block_size, n, cohesion_yx_block);
                    print_matrix(block_size, n, cohesion_xz_block);
                    print_matrix(block_size, n, cohesion_zx_block);
                    print_matrix(block_size, n, cohesion_yz_block);
                    print_matrix(block_size, n, cohesion_zy_block);
                    printf("(xb:%d, yb:%d, zb:%d)\n", xb, yb, zb);
                }
                time_start = omp_get_wtime();
                for (i = 0; i < x_block; ++i){
                    // size_xz = (xb == zb) ? i : block_size;
                    // size_yz = (yb == zb) ? i : block_size;
                    memcpy(distance_xz_block + i * block_size, D + zb + (xb + i) * n, sizeof(float)*z_block);
                }
                for (i = 0; i < y_block; ++i){
                    memcpy(distance_yz_block + i * block_size, D + zb + (yb + i) * n, sizeof(float)*z_block);
                }
                memops_loop_time += omp_get_wtime() - time_start;
                time_start = omp_get_wtime();
                conflict_xy_block = conflict_matrix + yb + xb * n;
                conflict_xz_block = conflict_matrix + zb + xb * n;
                conflict_yz_block = conflict_matrix + zb + yb * n;

                cohesion_xy_block = C + yb + xb * n;
                cohesion_yx_block = C + xb + yb * n;
                cohesion_xz_block = C + zb + xb * n;
                cohesion_zx_block = C + xb + zb * n;
                cohesion_yz_block = C + zb + yb * n;
                cohesion_zy_block = C + yb + zb * n;
                xend = x_block;
                ystart = 0;
                zstart = 0;
                if(xb == yb && yb == zb){
                    xend = x_block - 1;
                }
                for(x = 0; x < xend; ++x){
                    if(xb == yb){
                        ystart = x + 1;
                        conflict_yz_block += ystart*n;
                        cohesion_yz_block += ystart*n;
                        cohesion_yx_block += ystart*n;
                    }
                    for(y = ystart; y < y_block; ++y){
                        xy_reduction = 0.f;
                        yx_reduction = 0.f;
                        if(yb == zb){
                            zstart = y + 1;
                            cohesion_zx_block += zstart*n;
                            cohesion_zy_block += zstart*n;
                        }
                        for(z = zstart; z < z_block; ++z){
                            // update cohesion matrix blocks.
                            if(print_out)
                                printf("(x:%d, y:%d, z:%d) : (%.2f, %.2f, %.2f)\n", x, y, z, distance_xy_block[y + x * block_size], distance_xz_block[z + x * block_size], distance_yz_block[z + y * block_size]);
                            if(distance_xy_block[y + x * block_size] < distance_xz_block[z + x * block_size] && distance_xy_block[y + x * block_size] < distance_yz_block[z + y * block_size]){
                                yx_reduction += 1.f/conflict_xz_block[z];
                                xy_reduction += 1.f/conflict_yz_block[z];
                            }
                            else if(distance_xz_block[z + x * block_size] < distance_xy_block[y + x * block_size] && distance_xz_block[z + x * block_size] < distance_yz_block[z + y * block_size]){
                                cohesion_zx_block[x] += 1.f/conflict_xy_block[y];
                                cohesion_xz_block[z] += 1.f/conflict_yz_block[z];
                            }
                            else if(distance_yz_block[z + y * block_size] < distance_xz_block[z + x * block_size] && distance_yz_block[z + y * block_size] < distance_xy_block[y + x * block_size]){
                                cohesion_zy_block[y] += 1.f/conflict_xy_block[y];
                                cohesion_yz_block[z] += 1.f/conflict_xz_block[z];
                            }
                            // else{
                            //     if (distance_xy_block[y + x * block_size] == distance_xz_block[z + x * block_size]){
                            //         yx_reduction += 1.f/conflict_xz_block[z];
                            //         xy_reduction += .5f/conflict_yz_block[z];
                            //         cohesion_xz_block[z] += .5f/conflict_yz_block[z];
                            //         cohesion_zx_block[x] += 1.f/conflict_xy_block[y];
                            //     }

                            //     if (distance_xz_block[z + x * block_size] == distance_yz_block[z + y * block_size]){
                            //         cohesion_zx_block[x] += .5f / conflict_xy_block[y];
                            //         cohesion_zy_block[y] += .5f / conflict_xy_block[y];

                            //         cohesion_xz_block[z] += 1.f / conflict_yz_block[z];
                            //         cohesion_yz_block[z] += 1.f / conflict_xz_block[z];
                            //     }

                            //     if (distance_xy_block[y + x * block_size] == distance_yz_block[z + y * block_size]){
                            //         yx_reduction += .5f/conflict_xz_block[z];
                            //         xy_reduction += 1.f/conflict_yz_block[z];

                            //         cohesion_zy_block[y] += 1.f/conflict_xy_block[y];
                            //         cohesion_yz_block[z] += .5f/conflict_xz_block[z];
                            //     }
                            // }
                            cohesion_zx_block += n;
                            cohesion_zy_block += n;
                        }
                        conflict_yz_block += n;

                        cohesion_xy_block[y] += xy_reduction;
                        cohesion_yx_block[x] += yx_reduction;
                        cohesion_yx_block += n;
                        cohesion_yz_block += n;
                        cohesion_zx_block = C + xb + zb * n;
                        cohesion_zy_block = C + yb + zb * n;
                    }
                    conflict_xz_block += n;
                    conflict_xy_block += n;
                    conflict_yz_block = conflict_matrix + zb + yb * n;

                    cohesion_xz_block += n;
                    cohesion_xy_block += n;
                    cohesion_yx_block = C + xb + yb * n;
                    cohesion_yz_block = C + zb + yb * n;
                }
                if(print_out)
                    print_matrix(n, n, C);
                // if(iters == 15)
                //     exit(-1);
                iters++;
                cohesion_loop_time += omp_get_wtime() - time_start;
            }
        }
    }


    printf("========================================================\n");
    printf("Seq. Triplet Blocked Non-Powers of Two Loop Times\n");
    printf("========================================================\n");

    printf("memops loop time: %.5fs\n", memops_loop_time);
    printf("conflict focus size loop time: %.5fs\n", conflict_loop_time);
    printf("cohesion matrix update loop time: %.5fs\n\n", cohesion_loop_time);

    _mm_free(conflict_matrix);
}

void pald_triplet_intrin_powersoftwo(float *D, float beta, int n, float *C, int block_size){
    //TODO: Optimized sequential triplet code.
    float* restrict conflict_matrix = (float *)  _mm_malloc(n * n * sizeof(float), VECALIGN);
    // memset(conflict_matrix, 0, n * n * sizeof(float));
    unsigned int* restrict conflict_matrix_int = (unsigned int*)  _mm_malloc(n * n * sizeof(unsigned int), VECALIGN);
    // memset(conflict_matrix_int, 0, n * n * sizeof(unsigned int));

    float* restrict distance_xy_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
    float* restrict distance_xz_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
    float* restrict distance_yz_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);

    // float* restrict mask_xy_closest = (float *) _mm_malloc(block_size * sizeof(float), VECALIGN);
    // float* restrict mask_xz_closest = (float *) _mm_malloc(block_size * sizeof(float), VECALIGN);
    // float* restrict mask_yz_closest = (float *) _mm_malloc(block_size * sizeof(float), VECALIGN);

    // unsigned int* restrict mask_xy_closest_int = (unsigned int*) _mm_malloc(block_size * sizeof(unsigned int), VECALIGN);
    // unsigned int* restrict mask_xz_closest_int = (unsigned int*) _mm_malloc(block_size * sizeof(unsigned int), VECALIGN);
    // unsigned int* restrict mask_yz_closest_int = (unsigned int*) _mm_malloc(block_size * sizeof(unsigned int), VECALIGN);

    unsigned int scalar_xy_closest_int, scalar_xz_closest_int, scalar_yz_closest_int;

    unsigned int* restrict buffer_conflict_xz_block_int = (unsigned int *) _mm_malloc(block_size * block_size * sizeof(unsigned int), VECALIGN);
    unsigned int* restrict buffer_conflict_yz_block_int = (unsigned int *) _mm_malloc(block_size * block_size * sizeof(unsigned int), VECALIGN);
    unsigned int* restrict buffer_conflict_xy_block_int = (unsigned int *) _mm_malloc(block_size * block_size * sizeof(unsigned int), VECALIGN);

    // float* restrict buffer_contains_tie = (float *) _mm_malloc(block_size * sizeof(float), VECALIGN);

    // char distance_check_1 = 0;
    // char distance_check_2 = 0;
    // char distance_check_3 = 0;
    unsigned int distance_check_1_mask, distance_check_2_mask;
    unsigned int xy_reduction_int;
    float dist_xy  = 0.f;
    float conflict_xy_val = 0.f;
    unsigned int loop_len = 0;

    unsigned int *conflict_xy_block_int, *conflict_xz_block_int, *conflict_yz_block_int;
    // char print_out = 0;
    double time_start = 0.0, time_start2 = 0.0;
    double memops_loop_time = 0.0, conflict_loop_time = 0.0, cohesion_loop_time = 0.0;
    int xb, yb, zb, x, y, z;
    int i, j, k;
    // int size_xy = block_size, size_xz = block_size, size_yz = block_size;
    int xend, ystart, zstart;
    float xy_reduction = 0.f, yx_reduction = 0.f, cohesion_sum = 0.f;
    // compute conflict focus sizes.
    int iters = 0;
    int idx;
    time_start = omp_get_wtime();
    for (i = 0; i < n; ++i){
        for (j = i + 1; j < n; ++j){
            // conflict_matrix[j + i * n] = 2.;
            conflict_matrix_int[j + i * n] = 2;
        }
    }
    conflict_loop_time += omp_get_wtime() - time_start;
    // if(print_out)
    //     print_matrix(n, n, conflict_matrix);


    //TODO: Add another level of blocking.
    for(xb = 0; xb < n; xb += block_size){
        for(yb = xb; yb < n; yb += block_size){
            time_start = omp_get_wtime();
            for (i = 0; i < block_size; ++i){
                memcpy(distance_xy_block + i * block_size, D + yb + (xb + i) * n, sizeof(float)*block_size);
            }

            // memset(buffer_conflict_xy_block, 0, sizeof(float)*block_size*block_size);
            memset(buffer_conflict_xy_block_int, 0, sizeof(int)*block_size*block_size);
            memops_loop_time += omp_get_wtime() - time_start;

            // copy DXY block from D.
            for(zb = yb; zb < n; zb += block_size){
                //copy DXZ and DYZ blocks from D.
                time_start = omp_get_wtime();

                for (i = 0; i < block_size; ++i){
                    memcpy(distance_xz_block + i * block_size, D + zb + (xb + i) * n, sizeof(float)*block_size);
                }
                for(i = 0; i < block_size; ++i){
                    memcpy(distance_yz_block + i * block_size, D + zb + (yb + i) * n, sizeof(float)*block_size);
                }

                // memset(buffer_conflict_xz_block, 0, sizeof(float)*block_size*block_size);
                // memset(buffer_conflict_yz_block, 0, sizeof(float)*block_size*block_size);
                memset(buffer_conflict_xz_block_int, 0, sizeof(int)*block_size*block_size);
                memset(buffer_conflict_yz_block_int, 0, sizeof(int)*block_size*block_size);
                memops_loop_time += omp_get_wtime() - time_start;

                time_start = omp_get_wtime();

                xend = (xb == yb && yb == zb) ? block_size - 1 : block_size;
                // ystart = 0;
                // zstart = 0;
                // if(xb == yb && yb == zb){
                //     xend = block_size - 1;
                // }
                for(x = 0; x < xend; ++x){
                    // if(xb == yb){
                    //     ystart = x + 1;
                    //     // conflict_yz_block += ystart*n;
                    // }
                    ystart = (xb == yb) ? x + 1 : 0;
                    if(xb == yb){
                        for(y = x + 1; y < block_size; ++y){
                            // if(yb == zb){
                            //     zstart = y + 1;
                            // }
                            // xy_reduction = 0.f;
                            xy_reduction_int = 0;
                            zstart = (yb == zb) ? y + 1 : 0;
                            dist_xy = distance_xy_block[y + x * block_size];
                            // contains_tie = 0.f;
                            loop_len = block_size - zstart;
                            if(yb == zb){
                                // for (z = y + 1; z < block_size; ++z){
                                #pragma unroll(16)
                                for (z = 0; z < loop_len; ++z){
                                    idx = z + y + 1;
                                    //compute masks for conflict blocks.

                                    distance_check_1_mask = dist_xy < distance_xz_block[idx + x * block_size];
                                    distance_check_2_mask = dist_xy < distance_yz_block[idx + y * block_size];
                                    scalar_xy_closest_int = distance_check_1_mask & distance_check_2_mask;

                                    distance_check_1_mask = distance_xz_block[idx + x * block_size] < dist_xy;
                                    distance_check_2_mask =  distance_xz_block[idx + x * block_size] < distance_yz_block[idx + y * block_size];
                                    scalar_xz_closest_int = distance_check_1_mask & distance_check_2_mask;

                                    distance_check_1_mask = distance_yz_block[idx + y * block_size] < distance_xz_block[idx + x * block_size];
                                    distance_check_2_mask = distance_yz_block[idx + y * block_size] < dist_xy;
                                    scalar_yz_closest_int = distance_check_1_mask & distance_check_2_mask;


                                    xy_reduction_int += scalar_xz_closest_int + scalar_yz_closest_int;
                                    buffer_conflict_yz_block_int[idx + y * block_size] += scalar_xy_closest_int + scalar_xz_closest_int;
                                    buffer_conflict_xz_block_int[idx + x * block_size] += scalar_xy_closest_int + scalar_yz_closest_int;
                                }

                                buffer_conflict_xy_block_int[y + x * block_size] += xy_reduction_int;
                            }
                            else{
                                // _mm512_store_epi32(buffer_conflict_xy_block_int + y + x * block_size, conflict_xy);
                                // __m512 dist_xy_avx = _mm512_set1_ps(dist_xy);
                                // __m512i all_ones = _mm512_set1_epi32(1);
                                // __m512 dist_xz_avx, dist_yz_avx;
                                // __mmask16 cmp_result_1, cmp_result_2, cmp_result_3;
                                // __m512i conf_xy, conf_xz, conf_yz;
                                // // for(z = 0; z < block_size; z+=16){
                                // //     dist_xz_avx = _mm512_load_ps(distance_xz_block + z + x * block_size);
                                // //     dist_yz_avx = _mm512_load_ps(distance_yz_block + z + y * block_size);
                                // //     distance_check_1_mask = _mm512_cmplt_ps_mask(dist_xy_avx, dist_xz_avx);
                                // //     distance_check_1_mask = _mm512_cmplt_ps_mask(dist_xy_avx, dist_yz_avx);
                                // //     cmp_result_1 = distance_check_1_mask & distance_check_2_mask;

                                // //     distance_check_1_mask = _mm512_cmplt_ps_mask(dist_xz_avx, dist_xy_avx);
                                // //     distance_check_1_mask = _mm512_cmplt_ps_mask(dist_xz_avx, dist_yz_avx);
                                // //     cmp_result_2 = distance_check_1_mask & distance_check_2_mask;

                                // //     distance_check_1_mask = _mm512_cmplt_ps_mask(dist_yz_avx, dist_xy_avx);
                                // //     distance_check_1_mask = _mm512_cmplt_ps_mask(dist_yz_avx, dist_xz_avx);
                                // //     cmp_result_3 = distance_check_1_mask & distance_check_2_mask;

                                // // }
                                #pragma unroll(16)
                                for (z = 0; z < block_size; ++z){
                                    //compute masks for conflict blocks.
                                    distance_check_1_mask = dist_xy < distance_xz_block[z + x * block_size];
                                    distance_check_2_mask = dist_xy < distance_yz_block[z + y * block_size];
                                    scalar_xy_closest_int = distance_check_1_mask & distance_check_2_mask;

                                    distance_check_1_mask = distance_xz_block[z + x * block_size] < dist_xy;
                                    distance_check_2_mask =  distance_xz_block[z + x * block_size] < distance_yz_block[z + y * block_size];
                                    scalar_xz_closest_int = distance_check_1_mask & distance_check_2_mask;

                                    distance_check_1_mask = distance_yz_block[z + y * block_size] < distance_xz_block[z + x * block_size];
                                    distance_check_2_mask = distance_yz_block[z + y * block_size] < dist_xy;
                                    scalar_yz_closest_int = distance_check_1_mask & distance_check_2_mask;

                                    xy_reduction_int += scalar_xz_closest_int + scalar_yz_closest_int;
                                    buffer_conflict_yz_block_int[z + y * block_size] += scalar_xy_closest_int + scalar_xz_closest_int;
                                    buffer_conflict_xz_block_int[z + x * block_size] += scalar_xy_closest_int + scalar_yz_closest_int;
                                }
                                buffer_conflict_xy_block_int[y + x * block_size] += xy_reduction_int;
                            }
                            // conflict_yz_block += n;
                        }
                    }
                    else{
                        for(y = 0; y < block_size; ++y){
                            // if(yb == zb){
                            //     zstart = y + 1;
                            // }
                            // xy_reduction = 0.f;
                            xy_reduction_int = 0;
                            zstart = (yb == zb) ? y + 1 : 0;
                            dist_xy = distance_xy_block[y + x * block_size];
                            // contains_tie = 0.f;
                            loop_len = block_size - zstart;
                            if(yb == zb){
                                #pragma unroll(16)
                                for (z = 0; z < loop_len; ++z){
                                    idx = z + y + 1;
                                    //compute masks for conflict blocks.

                                    distance_check_1_mask = dist_xy < distance_xz_block[idx + x * block_size];
                                    distance_check_2_mask = dist_xy < distance_yz_block[idx + y * block_size];
                                    scalar_xy_closest_int = distance_check_1_mask & distance_check_2_mask;

                                    distance_check_1_mask = distance_xz_block[idx + x * block_size] < dist_xy;
                                    distance_check_2_mask =  distance_xz_block[idx + x * block_size] < distance_yz_block[idx + y * block_size];
                                    scalar_xz_closest_int = distance_check_1_mask & distance_check_2_mask;

                                    distance_check_1_mask = distance_yz_block[idx + y * block_size] < distance_xz_block[idx + x * block_size];
                                    distance_check_2_mask = distance_yz_block[idx + y * block_size] < dist_xy;
                                    scalar_yz_closest_int = distance_check_1_mask & distance_check_2_mask;


                                    xy_reduction_int += scalar_xz_closest_int + scalar_yz_closest_int;
                                    buffer_conflict_yz_block_int[idx + y * block_size] += scalar_xy_closest_int + scalar_xz_closest_int;
                                    buffer_conflict_xz_block_int[idx + x * block_size] += scalar_xy_closest_int + scalar_yz_closest_int;
                                }
                                buffer_conflict_xy_block_int[y + x * block_size] += xy_reduction_int;
                            }
                            else{
                                #pragma unroll(16)
                                for (z = 0; z < block_size; ++z){
                                    //compute masks for conflict blocks.
                                    distance_check_1_mask = dist_xy < distance_xz_block[z + x * block_size];
                                    distance_check_2_mask = dist_xy < distance_yz_block[z + y * block_size];
                                    scalar_xy_closest_int = distance_check_1_mask & distance_check_2_mask;

                                    distance_check_1_mask = distance_xz_block[z + x * block_size] < dist_xy;
                                    distance_check_2_mask =  distance_xz_block[z + x * block_size] < distance_yz_block[z + y * block_size];
                                    scalar_xz_closest_int = distance_check_1_mask & distance_check_2_mask;

                                    distance_check_1_mask = distance_yz_block[z + y * block_size] < distance_xz_block[z + x * block_size];
                                    distance_check_2_mask = distance_yz_block[z + y * block_size] < dist_xy;
                                    scalar_yz_closest_int = distance_check_1_mask & distance_check_2_mask;
                                    xy_reduction_int += scalar_xz_closest_int + scalar_yz_closest_int;
                                    buffer_conflict_yz_block_int[z + y * block_size] += scalar_xy_closest_int + scalar_xz_closest_int;
                                    buffer_conflict_xz_block_int[z + x * block_size] += scalar_xy_closest_int + scalar_yz_closest_int;
                                }
                                buffer_conflict_xy_block_int[y + x * block_size] += xy_reduction_int;
                            }
                            // conflict_yz_block += n;
                        }
                    }

                }
                conflict_loop_time += omp_get_wtime() - time_start;
                time_start2 = omp_get_wtime();
                // conflict_xy_block = conflict_matrix + yb + xb * n;
                // conflict_xz_block = conflict_matrix + zb + xb * n;
                // conflict_yz_block = conflict_matrix + zb + yb * n;
                conflict_xz_block_int = conflict_matrix_int + zb + xb * n;
                conflict_yz_block_int = conflict_matrix_int + zb + yb * n;

                for(i = 0; i < block_size; ++i){
                    for(j = 0; j < block_size; ++j){
                        // conflict_xz_block[j + i * n] += buffer_conflict_xz_block[j + i * block_size];
                        // conflict_yz_block[j + i * n] += buffer_conflict_yz_block[j + i * block_size];
                        conflict_xz_block_int[j + i * n] += buffer_conflict_xz_block_int[j + i * block_size];
                        conflict_yz_block_int[j + i * n] += buffer_conflict_yz_block_int[j + i * block_size];
                    }

                }
                memops_loop_time += omp_get_wtime() - time_start2;
                // printf("(xb: %d, yb: %d, zb: %d)\n", xb, yb, zb);
                // print_matrix_int(block_size, block_size, buffer_conflict_yz_block_int);
                // printf("[\n");
                // conflict_xy_block_int = conflict_matrix_int + yb + xb * n;
                // for(i = 0; i < block_size; ++i){
                //     for(j = 0; j < block_size; ++j){
                //         printf("%d ", conflict_xy_block_int[j + i * n] + buffer_conflict_xy_block_int[j + i * block_size]);
                //     }
                //     printf(";\n");
                // }
                // printf("];\n");

            }
            time_start2 = omp_get_wtime();
            // conflict_xy_block = conflict_matrix + yb + xb * n;
            conflict_xy_block_int = conflict_matrix_int + yb + xb * n;
            for(i = 0; i < block_size; ++i){
                for(j = 0; j < block_size; ++j){
                    // conflict_xy_block[j + i * n] += buffer_conflict_xy_block[j + i * block_size];
                    conflict_xy_block_int[j + i * n] += buffer_conflict_xy_block_int[j + i * block_size];
                }
                // conflict_xy_block += n;
            }
            // print_matrix_int(n,n, conflict_matrix_int);
            // printf("(xb: %d, yb: %d)\n", xb, yb);
            // print_matrix_int(block_size, block_size, buffer_conflict_xy_block_int);
            memops_loop_time += omp_get_wtime() - time_start2;
        }
    }
    time_start = omp_get_wtime();
    for(i = 0; i < n * n; ++i){
        // conflict_matrix[i] = 1.f/conflict_matrix[i];
        conflict_matrix[i] = 1.f/conflict_matrix_int[i];
    }
    // print_matrix_int(n, n, conflict_matrix_int);
    conflict_loop_time += omp_get_wtime() - time_start;
    // return;
    // printf("\n\n");
        // initialize diagonal of C.
    float sum;
    time_start = omp_get_wtime();
    for (i = 0; i < n; ++i){
        sum = 0.f;
        for (j = 0; j < i; ++j){
            sum += conflict_matrix[i + j * n];
        }
        for (j = i + 1; j < n; ++j){
            sum += conflict_matrix[j + i * n];
        }
        C[i + i * n] = sum;
    }
    cohesion_loop_time += omp_get_wtime() - time_start;
    iters = 0;

    // print_matrix_int(n, n, conflict_matrix_int);
    time_start = omp_get_wtime();
    _mm_free(conflict_matrix_int);
    _mm_free(buffer_conflict_xy_block_int); _mm_free(buffer_conflict_xz_block_int); _mm_free(buffer_conflict_yz_block_int);
    _mm_free(distance_xy_block); _mm_free(distance_xz_block); _mm_free(distance_yz_block);
    float *conflict_xy_block, *conflict_xz_block, *conflict_yz_block;
    float *cohesion_xy_block ;
    float *cohesion_yx_block;
    float *cohesion_xz_block;
    float *cohesion_zx_block;
    float *cohesion_yz_block;
    float *cohesion_zy_block;
    block_size/=2;

    // float* restrict mask_xy_closest = (float *) _mm_malloc(block_size * sizeof(float), VECALIGN);
    // float* restrict mask_xz_closest = (float *) _mm_malloc(block_size * sizeof(float), VECALIGN);
    // float* restrict mask_yz_closest = (float *) _mm_malloc(block_size * sizeof(float), VECALIGN);

    float scalar_xy_closest, scalar_xz_closest, scalar_yz_closest;

    // float* restrict mask_tie_xy_xz = (float *) _mm_malloc(block_size * sizeof(float), VECALIGN);
    // float* restrict mask_tie_xy_yz = (float *) _mm_malloc(block_size * sizeof(float), VECALIGN);
    // float* restrict mask_tie_xz_yz = (float *) _mm_malloc(block_size * sizeof(float), VECALIGN);

    float* buffer_conflict_xz_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
    float* buffer_conflict_yz_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
    float* buffer_conflict_xy_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);

    float* restrict buffer_zx_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
    float* restrict buffer_zy_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
    float* restrict buffer_xz_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
    float* restrict buffer_yz_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
    float* restrict buffer_xy_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
    float* restrict buffer_yx_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);

    distance_xy_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
    distance_xz_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
    distance_yz_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);

    float* tmp_buffer_conflict_yz_block = buffer_conflict_yz_block;
    float* tmp_distance_yz_block = distance_yz_block;
    float* tmp_buffer_yz_block = buffer_yz_block;
    float* tmp_buffer_zy_block = buffer_zy_block;
    memops_loop_time += omp_get_wtime() - time_start;

    for(xb = 0; xb < n; xb += block_size){
        for(yb = xb; yb < n; yb += block_size){
            time_start = omp_get_wtime();
            for (i = 0; i < block_size; ++i){
                memcpy(distance_xy_block + i * block_size, D + yb + (xb + i) * n, sizeof(float)*block_size);
            }

            memset(buffer_yx_block,0,sizeof(float)*block_size*block_size);
            memset(buffer_xy_block,0,sizeof(float)*block_size*block_size);
            memops_loop_time += omp_get_wtime() - time_start;
            for(zb = yb; zb < n; zb += block_size){
                time_start = omp_get_wtime();
                conflict_xy_block = conflict_matrix + yb + xb * n;
                conflict_xz_block = conflict_matrix + zb + xb * n;
                conflict_yz_block = conflict_matrix + zb + yb * n;
                if(xb == yb){
                    #pragma unroll(8)
                    for (i = 0; i < block_size; ++i){
                        memcpy(distance_xz_block + i * block_size, D + zb + (xb + i) * n, sizeof(float)*block_size);
                        memcpy(buffer_conflict_xz_block + i * block_size, conflict_xz_block + i * n, sizeof(float)*block_size);
                        // memcpy(distance_yz_block + i * block_size, D + zb + (yb + i) * n, sizeof(float)*block_size);
                        // memcpy(buffer_conflict_yz_block + i * block_size, conflict_yz_block + i * n, sizeof(float)*block_size);
                    }
                    distance_yz_block = distance_xz_block;
                    buffer_conflict_yz_block = buffer_conflict_xz_block;
                    memset(buffer_zx_block,0,sizeof(float)*block_size*block_size);
                    memset(buffer_xz_block,0,sizeof(float)*block_size*block_size);

                    buffer_zy_block = buffer_zx_block;
                    buffer_yz_block = buffer_xz_block;
                }
                else{
                    distance_yz_block = tmp_distance_yz_block;
                    buffer_conflict_yz_block = tmp_buffer_conflict_yz_block;
                    buffer_zy_block = tmp_buffer_zy_block;
                    buffer_yz_block = tmp_buffer_yz_block;
                    #pragma unroll(8)
                    for (i = 0; i < block_size; ++i){
                        memcpy(distance_xz_block + i * block_size, D + zb + (xb + i) * n, sizeof(float)*block_size);
                        memcpy(buffer_conflict_xz_block + i * block_size, conflict_xz_block + i * n, sizeof(float)*block_size);
                        memcpy(distance_yz_block + i * block_size, D + zb + (yb + i) * n, sizeof(float)*block_size);
                        memcpy(buffer_conflict_yz_block + i * block_size, conflict_yz_block + i * n, sizeof(float)*block_size);
                    }
                    memset(buffer_zx_block,0,sizeof(float)*block_size*block_size);
                    memset(buffer_zy_block,0,sizeof(float)*block_size*block_size);
                    memset(buffer_xz_block,0,sizeof(float)*block_size*block_size);
                    memset(buffer_yz_block,0,sizeof(float)*block_size*block_size);
                }
                memops_loop_time += omp_get_wtime() - time_start;

                time_start = omp_get_wtime();
                xend = block_size;
                // ystart = 0;
                // zstart = 0;
                if(xb == yb && yb == zb){
                    xend = block_size - 1;
                }
                for(x = 0; x < xend; ++x){
                    // if(xb == yb){
                    //     ystart = x + 1;

                    // }
                    ystart = (xb == yb) ? x + 1 : 0;
                    for(y = ystart; y < block_size; ++y){
                        xy_reduction = 0.f; yx_reduction = 0.f;
                        // if(yb == zb){
                        //     zstart = y + 1;
                        // }
                            // zstart = (yb == zb) ? y + 1 : 0;
                        dist_xy = distance_xy_block[y + x * block_size];
                        // loop_len = block_size - zstart;
                        if(yb == zb){
                            loop_len = block_size - y - 1;
                            conflict_xy_val = conflict_xy_block[y];
                            // for (z = y + 1; z < block_size; ++z){
                            for (z = 0; z < loop_len; ++z){
                                //compute masks for conflict blocks.
                                idx = z + y + 1;
                                distance_check_1_mask = dist_xy < distance_xz_block[idx + x * block_size];
                                distance_check_2_mask = dist_xy < distance_yz_block[idx + y * block_size];
                                scalar_xy_closest = distance_check_1_mask & distance_check_2_mask;

                                distance_check_1_mask = distance_xz_block[idx + x * block_size] < dist_xy;
                                distance_check_2_mask =  distance_xz_block[idx + x * block_size] < distance_yz_block[idx + y * block_size];
                                scalar_xz_closest = distance_check_1_mask & distance_check_2_mask;

                                distance_check_1_mask = distance_yz_block[idx + y * block_size] < distance_xz_block[idx + x * block_size];
                                distance_check_2_mask = distance_yz_block[idx + y * block_size] < dist_xy;
                                scalar_yz_closest = distance_check_1_mask & distance_check_2_mask;

                                // xy closest pair.
                                yx_reduction += scalar_xy_closest*buffer_conflict_xz_block[idx + x * block_size];
                                xy_reduction += scalar_xy_closest*buffer_conflict_yz_block[idx + y * block_size];

                                // xz closest pair.
                                buffer_xz_block[idx + x * block_size] += scalar_xz_closest*buffer_conflict_yz_block[idx + y * block_size];
                                buffer_zx_block[idx + x * block_size] += scalar_xz_closest*conflict_xy_val;

                                // yz closest pair.
                                buffer_yz_block[idx + y * block_size] += scalar_yz_closest*buffer_conflict_xz_block[idx + x * block_size];
                                buffer_zy_block[idx + y * block_size] += scalar_yz_closest*conflict_xy_val;
                            }
                            buffer_xy_block[y + x * block_size] += xy_reduction;
                            buffer_yx_block[y + x * block_size] += yx_reduction;
                        }
                        else{
                            conflict_xy_val = conflict_xy_block[y];
                            // __m512 all_ones = _mm512_set1_ps(1.f);
                            // __m512 dist_xy_avx = _mm512_set1_ps(dist_xy);
                            // __m512 conflict_xy_avx = _mm512_set1_ps(conflict_xy_val);

                            // __m512 xy_reduction_avx = _mm512_setzero();
                            // __m512 yx_reduction_avx = _mm512_setzero();
                            // __m512 dist_xz_avx, dist_yz_avx, conflict_xz_avx, conflict_yz_avx;
                            // __m512 cohesion_xz_avx, cohesion_zx_avx, cohesion_yz_avx, cohesion_zy_avx;
                            // __mmask16 mask_xy_closest, mask_xz_closest, mask_yz_closest;

                            // __m512 xy_reduction_avx_1 = _mm512_setzero();
                            // __m512 yx_reduction_avx_1 = _mm512_setzero();
                            // __m512 dist_xz_avx_1, dist_yz_avx_1, conflict_xz_avx_1, conflict_yz_avx_1;
                            // __m512 cohesion_xz_avx_1, cohesion_zx_avx_1, cohesion_yz_avx_1, cohesion_zy_avx_1;
                            // __mmask16 mask_xy_closest_1, mask_xz_closest_1, mask_yz_closest_1;

                            // __m512 xy_reduction_avx_2 = _mm512_setzero();
                            // __m512 yx_reduction_avx_2 = _mm512_setzero();
                            // __m512 dist_xz_avx_2, dist_yz_avx_2, conflict_xz_avx_2, conflict_yz_avx_2;
                            // __m512 cohesion_xz_avx_2, cohesion_zx_avx_2, cohesion_yz_avx_2, cohesion_zy_avx_2;
                            // __mmask16 mask_xy_closest_2, mask_xz_closest_2, mask_yz_closest_2;

                            // __m512 xy_reduction_avx_3 = _mm512_setzero();
                            // __m512 yx_reduction_avx_3 = _mm512_setzero();
                            // __m512 dist_xz_avx_3, dist_yz_avx_3, conflict_xz_avx_3, conflict_yz_avx_3;
                            // __m512 cohesion_xz_avx_3, cohesion_zx_avx_3, cohesion_yz_avx_3, cohesion_zy_avx_3;
                            // __mmask16 mask_xy_closest_3, mask_xz_closest_3, mask_yz_closest_3;

                            // __mmask16 distance_check_3_mask, distance_check_4_mask;
                            // __mmask16 distance_check_5_mask, distance_check_6_mask;
                            // __mmask16 distance_check_7_mask, distance_check_8_mask;

                            // for(z = 0; z < block_size; z += 128){
                            //     dist_xz_avx = _mm512_load_ps(distance_xz_block + z + x * block_size);
                            //     dist_yz_avx = _mm512_load_ps(distance_yz_block + z + y * block_size);
                            //     cohesion_xz_avx = _mm512_load_ps(buffer_xz_block + z + x * block_size);
                            //     cohesion_zx_avx = _mm512_load_ps(buffer_zx_block + z + x * block_size);
                            //     cohesion_yz_avx = _mm512_load_ps(buffer_yz_block + z + y * block_size);
                            //     cohesion_zy_avx = _mm512_load_ps(buffer_zy_block + z + y * block_size);
                            //     conflict_yz_avx = _mm512_load_ps(buffer_conflict_yz_block + z + y * block_size);
                            //     conflict_xz_avx = _mm512_load_ps(buffer_conflict_xz_block + z + x * block_size);

                            //     dist_xz_avx_1 = _mm512_load_ps(distance_xz_block + z + 16 + x * block_size);
                            //     dist_yz_avx_1 = _mm512_load_ps(distance_yz_block + z + 16 + y * block_size);
                            //     cohesion_xz_avx_1 = _mm512_load_ps(buffer_xz_block + z + 16 + x * block_size);
                            //     cohesion_zx_avx_1 = _mm512_load_ps(buffer_zx_block + z + 16 + x * block_size);
                            //     cohesion_yz_avx_1 = _mm512_load_ps(buffer_yz_block + z + 16 + y * block_size);
                            //     cohesion_zy_avx_1 = _mm512_load_ps(buffer_zy_block + z + 16 + y * block_size);
                            //     conflict_yz_avx_1 = _mm512_load_ps(buffer_conflict_yz_block + z + 16 + y * block_size);
                            //     conflict_xz_avx_1 = _mm512_load_ps(buffer_conflict_xz_block + z + 16 + x * block_size);

                            //     dist_xz_avx_2 = _mm512_load_ps(distance_xz_block + z + 32 + x * block_size);
                            //     dist_yz_avx_2 = _mm512_load_ps(distance_yz_block + z + 32 + y * block_size);
                            //     cohesion_xz_avx_2 = _mm512_load_ps(buffer_xz_block + z + 32 + x * block_size);
                            //     cohesion_zx_avx_2 = _mm512_load_ps(buffer_zx_block + z + 32 + x * block_size);
                            //     cohesion_yz_avx_2 = _mm512_load_ps(buffer_yz_block + z + 32 + y * block_size);
                            //     cohesion_zy_avx_2 = _mm512_load_ps(buffer_zy_block + z + 32 + y * block_size);
                            //     conflict_yz_avx_2 = _mm512_load_ps(buffer_conflict_yz_block + z + 32 + y * block_size);
                            //     conflict_xz_avx_2 = _mm512_load_ps(buffer_conflict_xz_block + z + 32 + x * block_size);

                            //     dist_xz_avx_3 = _mm512_load_ps(distance_xz_block + z + 64 + x * block_size);
                            //     dist_yz_avx_3 = _mm512_load_ps(distance_yz_block + z + 64 + y * block_size);
                            //     cohesion_xz_avx_3 = _mm512_load_ps(buffer_xz_block + z + 64 + x * block_size);
                            //     cohesion_zx_avx_3 = _mm512_load_ps(buffer_zx_block + z + 64 + x * block_size);
                            //     cohesion_yz_avx_3 = _mm512_load_ps(buffer_yz_block + z + 64 + y * block_size);
                            //     cohesion_zy_avx_3 = _mm512_load_ps(buffer_zy_block + z + 64 + y * block_size);
                            //     conflict_yz_avx_3 = _mm512_load_ps(buffer_conflict_yz_block + z + 64 + y * block_size);
                            //     conflict_xz_avx_3 = _mm512_load_ps(buffer_conflict_xz_block + z + 64 + x * block_size);

                            //     distance_check_1_mask = _mm512_cmplt_ps_mask(dist_xy_avx, dist_xz_avx);
                            //     distance_check_2_mask = _mm512_cmplt_ps_mask(dist_xy_avx, dist_yz_avx);
                            //     mask_xy_closest = distance_check_1_mask & distance_check_2_mask;

                            //     distance_check_1_mask = _mm512_cmplt_ps_mask(dist_xz_avx, dist_xy_avx);
                            //     distance_check_2_mask = _mm512_cmplt_ps_mask(dist_xz_avx, dist_yz_avx);
                            //     mask_xz_closest = distance_check_1_mask & distance_check_2_mask;

                            //     distance_check_1_mask = _mm512_cmplt_ps_mask(dist_yz_avx, dist_xz_avx);
                            //     distance_check_2_mask = _mm512_cmplt_ps_mask(dist_yz_avx, dist_xy_avx);
                            //     mask_yz_closest = distance_check_1_mask & distance_check_2_mask;

                            //     distance_check_3_mask = _mm512_cmplt_ps_mask(dist_xy_avx, dist_xz_avx);
                            //     distance_check_4_mask = _mm512_cmplt_ps_mask(dist_xy_avx, dist_yz_avx);
                            //     mask_xy_closest_1 = distance_check_3_mask & distance_check_4_mask;

                            //     distance_check_3_mask = _mm512_cmplt_ps_mask(dist_xz_avx, dist_xy_avx);
                            //     distance_check_4_mask = _mm512_cmplt_ps_mask(dist_xz_avx, dist_yz_avx);
                            //     mask_xz_closest_1 = distance_check_3_mask & distance_check_4_mask;

                            //     distance_check_3_mask = _mm512_cmplt_ps_mask(dist_yz_avx, dist_xz_avx);
                            //     distance_check_4_mask = _mm512_cmplt_ps_mask(dist_yz_avx, dist_xy_avx);
                            //     mask_yz_closest_1 = distance_check_3_mask & distance_check_4_mask;

                            //     distance_check_5_mask = _mm512_cmplt_ps_mask(dist_xy_avx, dist_xz_avx);
                            //     distance_check_6_mask = _mm512_cmplt_ps_mask(dist_xy_avx, dist_yz_avx);
                            //     mask_xy_closest_2 = distance_check_5_mask & distance_check_6_mask;

                            //     distance_check_5_mask = _mm512_cmplt_ps_mask(dist_xz_avx, dist_xy_avx);
                            //     distance_check_6_mask = _mm512_cmplt_ps_mask(dist_xz_avx, dist_yz_avx);
                            //     mask_xz_closest_2 = distance_check_5_mask & distance_check_6_mask;

                            //     distance_check_5_mask = _mm512_cmplt_ps_mask(dist_yz_avx, dist_xz_avx);
                            //     distance_check_6_mask = _mm512_cmplt_ps_mask(dist_yz_avx, dist_xy_avx);
                            //     mask_yz_closest_2 = distance_check_5_mask & distance_check_6_mask;

                            //     distance_check_7_mask = _mm512_cmplt_ps_mask(dist_xy_avx, dist_xz_avx);
                            //     distance_check_8_mask = _mm512_cmplt_ps_mask(dist_xy_avx, dist_yz_avx);
                            //     mask_xy_closest_3 = distance_check_7_mask & distance_check_8_mask;

                            //     distance_check_7_mask = _mm512_cmplt_ps_mask(dist_xz_avx, dist_xy_avx);
                            //     distance_check_8_mask = _mm512_cmplt_ps_mask(dist_xz_avx, dist_yz_avx);
                            //     mask_xz_closest_3 = distance_check_7_mask & distance_check_8_mask;

                            //     distance_check_7_mask = _mm512_cmplt_ps_mask(dist_yz_avx, dist_xz_avx);
                            //     distance_check_8_mask = _mm512_cmplt_ps_mask(dist_yz_avx, dist_xy_avx);
                            //     mask_yz_closest_3 = distance_check_7_mask & distance_check_8_mask;


                            //     xy_reduction_avx = _mm512_mask_fmadd_ps(all_ones, mask_xy_closest, conflict_yz_avx, xy_reduction_avx);
                            //     yx_reduction_avx = _mm512_mask_fmadd_ps(all_ones, mask_xy_closest, conflict_xz_avx, yx_reduction_avx);

                            //     xy_reduction_avx_1 = _mm512_mask_fmadd_ps(all_ones, mask_xy_closest_1, conflict_yz_avx_1, xy_reduction_avx_1);
                            //     yx_reduction_avx_1 = _mm512_mask_fmadd_ps(all_ones, mask_xy_closest_1, conflict_xz_avx_1, yx_reduction_avx_1);

                            //     xy_reduction_avx_2 = _mm512_mask_fmadd_ps(all_ones, mask_xy_closest_2, conflict_yz_avx_2, xy_reduction_avx_2);
                            //     yx_reduction_avx_2 = _mm512_mask_fmadd_ps(all_ones, mask_xy_closest_2, conflict_xz_avx_2, yx_reduction_avx_2);

                            //     xy_reduction_avx_3 = _mm512_mask_fmadd_ps(all_ones, mask_xy_closest_3, conflict_yz_avx_3, xy_reduction_avx_3);
                            //     yx_reduction_avx_3 = _mm512_mask_fmadd_ps(all_ones, mask_xy_closest_3, conflict_xz_avx_3, yx_reduction_avx_3);

                            //     _mm512_store_ps(buffer_xz_block + z + x * block_size, _mm512_mask_fmadd_ps(all_ones, mask_xz_closest, conflict_yz_avx, cohesion_xz_avx));
                            //     _mm512_store_ps(buffer_zx_block + z + x * block_size, _mm512_mask_fmadd_ps(all_ones, mask_xz_closest, conflict_xy_avx, cohesion_zx_avx));
                            //     _mm512_store_ps(buffer_yz_block + z + y * block_size, _mm512_mask_fmadd_ps(all_ones, mask_yz_closest, conflict_xz_avx, cohesion_yz_avx));
                            //     _mm512_store_ps(buffer_zy_block + z + y * block_size, _mm512_mask_fmadd_ps(all_ones, mask_yz_closest, conflict_xy_avx, cohesion_zy_avx));

                            //     _mm512_store_ps(buffer_xz_block + z + 16 + x * block_size, _mm512_mask_fmadd_ps(all_ones, mask_xz_closest_1, conflict_yz_avx_1, cohesion_xz_avx_1));
                            //     _mm512_store_ps(buffer_zx_block + z + 16 + x * block_size, _mm512_mask_fmadd_ps(all_ones, mask_xz_closest_1, conflict_xy_avx, cohesion_zx_avx_1));
                            //     _mm512_store_ps(buffer_yz_block + z + 16 + y * block_size, _mm512_mask_fmadd_ps(all_ones, mask_yz_closest_1, conflict_xz_avx_1, cohesion_yz_avx_1));
                            //     _mm512_store_ps(buffer_zy_block + z + 16 + y * block_size, _mm512_mask_fmadd_ps(all_ones, mask_yz_closest_1, conflict_xy_avx, cohesion_zy_avx_1));

                            //     _mm512_store_ps(buffer_xz_block + z + 32 + x * block_size, _mm512_mask_fmadd_ps(all_ones, mask_xz_closest_2, conflict_yz_avx_2, cohesion_xz_avx_2));
                            //     _mm512_store_ps(buffer_zx_block + z + 32 + x * block_size, _mm512_mask_fmadd_ps(all_ones, mask_xz_closest_2, conflict_xy_avx, cohesion_zx_avx_2));
                            //     _mm512_store_ps(buffer_yz_block + z + 32 + y * block_size, _mm512_mask_fmadd_ps(all_ones, mask_yz_closest_2, conflict_xz_avx_2, cohesion_yz_avx_2));
                            //     _mm512_store_ps(buffer_zy_block + z + 32 + y * block_size, _mm512_mask_fmadd_ps(all_ones, mask_yz_closest_2, conflict_xy_avx, cohesion_zy_avx_2));

                            //     _mm512_store_ps(buffer_xz_block + z + 64 + x * block_size, _mm512_mask_fmadd_ps(all_ones, mask_xz_closest_3, conflict_yz_avx_3, cohesion_xz_avx_3));
                            //     _mm512_store_ps(buffer_zx_block + z + 64 + x * block_size, _mm512_mask_fmadd_ps(all_ones, mask_xz_closest_3, conflict_xy_avx, cohesion_zx_avx_3));
                            //     _mm512_store_ps(buffer_yz_block + z + 64 + y * block_size, _mm512_mask_fmadd_ps(all_ones, mask_yz_closest_3, conflict_xz_avx_3, cohesion_yz_avx_3));
                            //     _mm512_store_ps(buffer_zy_block + z + 64 + y * block_size, _mm512_mask_fmadd_ps(all_ones, mask_yz_closest_3, conflict_xy_avx, cohesion_zy_avx_3));
                            // }
                            // xy_reduction_avx += xy_reduction_avx_1 + xy_reduction_avx_2 + xy_reduction_avx_3;
                            // yx_reduction_avx += yx_reduction_avx_1 + yx_reduction_avx_2 + yx_reduction_avx_3;
                            // xy_reduction += _mm512_reduce_add_ps(xy_reduction_avx);
                            // yx_reduction += _mm512_reduce_add_ps(yx_reduction_avx);
                            #pragma unroll(8)
                            //update cohesion blocks.
                            for (z = 0; z < block_size; ++z){
                                // xy closest pair.
                                distance_check_1_mask = dist_xy < distance_xz_block[z + x * block_size];
                                distance_check_2_mask = dist_xy < distance_yz_block[z + y * block_size];
                                scalar_xy_closest = distance_check_1_mask & distance_check_2_mask;

                                xy_reduction += scalar_xy_closest*buffer_conflict_yz_block[z + y * block_size];
                                yx_reduction += scalar_xy_closest*buffer_conflict_xz_block[z + x * block_size];

                                // xz closest pair.
                                distance_check_1_mask = distance_xz_block[z + x * block_size] < dist_xy;
                                distance_check_2_mask =  distance_xz_block[z + x * block_size] < distance_yz_block[z + y * block_size];
                                scalar_xz_closest = distance_check_1_mask & distance_check_2_mask;

                                buffer_xz_block[z + x * block_size] += scalar_xz_closest*buffer_conflict_yz_block[z + y * block_size];
                                buffer_zx_block[z + x * block_size] += scalar_xz_closest*conflict_xy_val;

                                //yz closest pair.
                                distance_check_1_mask = distance_yz_block[z + y * block_size] < distance_xz_block[z + x * block_size];
                                distance_check_2_mask = distance_yz_block[z + y * block_size] < dist_xy;
                                scalar_yz_closest = distance_check_1_mask & distance_check_2_mask;

                                buffer_yz_block[z + y * block_size] += scalar_yz_closest*buffer_conflict_xz_block[z + x * block_size];
                                buffer_zy_block[z + y * block_size] += scalar_yz_closest*conflict_xy_val;
                            }
                            buffer_xy_block[y + x * block_size] += xy_reduction;
                            buffer_yx_block[y + x * block_size] += yx_reduction;
                        }
                    }
                    conflict_xy_block += n;
                }


                time_start2 = omp_get_wtime();
                if(xb == yb){
                    cohesion_zx_block = C + xb + zb * n;
                    cohesion_xz_block = C + zb + xb * n;
                    for(i = 0; i < block_size; ++i){
                        for(j = 0; j < block_size; ++j){
                            // printf("idx: %d\n", n*j + i);
                            cohesion_zx_block[j] += buffer_zx_block[i + j * block_size];
                            // cohesion_yx_block[j] += buffer_yx_block[i + j * block_size];
                        }
                        cohesion_zx_block += n;
                    }
                    for(i = 0; i < block_size; ++i){
                        for(j = 0; j < block_size; ++j){
                            cohesion_xz_block[j] += buffer_xz_block[j + i * block_size];
                            // cohesion_xy_block[j] += buffer_xy_block[j + i * block_size];
                        }
                        cohesion_xz_block += n;
                    }

                }
                else{
                    cohesion_zx_block = C + xb + zb * n;
                    cohesion_zy_block = C + yb + zb * n;
                    cohesion_xz_block = C + zb + xb * n;
                    cohesion_yz_block = C + zb + yb * n;
                    for(i = 0; i < block_size; ++i){
                        for(j = 0; j < block_size; ++j){
                            // printf("idx: %d\n", n*j + i);
                            cohesion_zx_block[j] += buffer_zx_block[i + j * block_size];
                            cohesion_zy_block[j] += buffer_zy_block[i + j * block_size];
                            // cohesion_yx_block[j] += buffer_yx_block[i + j * block_size];
                        }
                        cohesion_zx_block += n;
                        cohesion_zy_block += n;
                    }
                    for(i = 0; i < block_size; ++i){
                        for(j = 0; j < block_size; ++j){
                            cohesion_xz_block[j] += buffer_xz_block[j + i * block_size];
                            cohesion_yz_block[j] += buffer_yz_block[j + i * block_size];
                            // cohesion_xy_block[j] += buffer_xy_block[j + i * block_size];
                        }
                        cohesion_xz_block += n;
                        cohesion_yz_block += n;
                    }
                }

                memops_loop_time += omp_get_wtime() - time_start2;
                cohesion_loop_time += omp_get_wtime() - time_start;
            }
            time_start2 = omp_get_wtime();
            cohesion_xy_block = C + yb + xb * n;
            cohesion_yx_block = C + xb + yb * n;
            for(i = 0; i < block_size; ++i){
                for(j = 0; j < block_size; ++j){
                    // printf("idx: %d\n", n*j + i);
                    cohesion_yx_block[j] += buffer_yx_block[i + j * block_size];
                }
                cohesion_yx_block += n;
            }

            for(i = 0; i < block_size; ++i){
                for(j = 0; j < block_size; ++j){
                    cohesion_xy_block[j] += buffer_xy_block[j + i * block_size];
                }
                cohesion_xy_block += n;
            }
            memops_loop_time += omp_get_wtime() - time_start2;
        }
    }
    // print_matrix(n, n, C);

    printf("==============================================================\n");
    printf("Seq. Triplet Explict Intrinsics Powers of Two Loop Times\n");
    printf("==============================================================\n");

    printf("memops loop time: %.5fs\n", memops_loop_time);
    printf("conflict focus size loop time: %.5fs\n", conflict_loop_time);
    printf("cohesion matrix update loop time: %.5fs\n\n", cohesion_loop_time);

    _mm_free(distance_xy_block); _mm_free(distance_xz_block); _mm_free(tmp_distance_yz_block);
    // _mm_free(mask_tie_xy_xz); _mm_free(mask_tie_xy_yz); _mm_free(mask_tie_xz_yz);
    _mm_free(buffer_zx_block); _mm_free(tmp_buffer_zy_block); _mm_free(buffer_yx_block);
    _mm_free(buffer_xz_block); _mm_free(tmp_buffer_yz_block); _mm_free(buffer_xy_block);
    // _mm_free(mask_xy_closest); _mm_free(mask_xz_closest); _mm_free(mask_yz_closest);
    _mm_free(buffer_conflict_xz_block); _mm_free(tmp_buffer_conflict_yz_block); _mm_free(buffer_conflict_xy_block);
    _mm_free(conflict_matrix);
    // _mm_free(conflict_matrix_int);
    // _mm_free(mask_xy_closest_int); _mm_free(mask_xz_closest_int); _mm_free(mask_yz_closest_int);
    // _mm_free(buffer_conflict_xz_block_int); _mm_free(buffer_conflict_yz_block_int); _mm_free(buffer_conflict_xy_block_int);

}

void pald_triplet_intrin(float *D, float beta, unsigned int n, float *C, unsigned int block_size){
    //TODO: Optimized sequential triplet code.
    float* restrict conflict_matrix = (float *)  _mm_malloc(n * n * sizeof(float), VECALIGN);
    // memset(conflict_matrix, 0, n * n * sizeof(float));
    unsigned int* restrict conflict_matrix_int = (unsigned int*)  _mm_malloc(n * n * sizeof(unsigned int), VECALIGN);
    // memset(conflict_matrix_int, 0, n * n * sizeof(unsigned int));

    float* restrict distance_xy_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
    float* restrict distance_xz_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
    float* restrict distance_yz_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);

    // float* restrict mask_xy_closest = (float *) _mm_malloc(block_size * sizeof(float), VECALIGN);
    // float* restrict mask_xz_closest = (float *) _mm_malloc(block_size * sizeof(float), VECALIGN);
    // float* restrict mask_yz_closest = (float *) _mm_malloc(block_size * sizeof(float), VECALIGN);

    // unsigned int* restrict mask_xy_closest_int = (unsigned int*) _mm_malloc(block_size * sizeof(unsigned int), VECALIGN);
    // unsigned int* restrict mask_xz_closest_int = (unsigned int*) _mm_malloc(block_size * sizeof(unsigned int), VECALIGN);
    // unsigned int* restrict mask_yz_closest_int = (unsigned int*) _mm_malloc(block_size * sizeof(unsigned int), VECALIGN);

    unsigned int scalar_xy_closest_int, scalar_xz_closest_int, scalar_yz_closest_int;

    unsigned int* restrict buffer_conflict_xz_block_int = (unsigned int *) _mm_malloc(block_size * block_size * sizeof(unsigned int), VECALIGN);
    unsigned int* restrict buffer_conflict_yz_block_int = (unsigned int *) _mm_malloc(block_size * block_size * sizeof(unsigned int), VECALIGN);
    unsigned int* restrict buffer_conflict_xy_block_int = (unsigned int *) _mm_malloc(block_size * block_size * sizeof(unsigned int), VECALIGN);

    // float* restrict buffer_contains_tie = (float *) _mm_malloc(block_size * sizeof(float), VECALIGN);

    // char distance_check_1 = 0;
    // char distance_check_2 = 0;
    // char distance_check_3 = 0;
    unsigned int distance_check_1_mask, distance_check_2_mask;
    unsigned int xy_reduction_int;
    float dist_xy  = 0.f;
    float conflict_xy_val = 0.f;
    unsigned int loop_len = 0;

    unsigned int *conflict_xy_block_int, *conflict_xz_block_int, *conflict_yz_block_int;
    // char print_out = 0;
    double time_start = 0.0, time_start2 = 0.0;
    double memops_loop_time = 0.0, conflict_loop_time = 0.0, cohesion_loop_time = 0.0;
    unsigned int xb, yb, zb, x, y, z;
    unsigned int i, j, k;
    unsigned int x_block, y_block, z_block;
    // int size_xy = block_size, size_xz = block_size, size_yz = block_size;
    unsigned int xend, ystart, zstart;
    float xy_reduction = 0.f, yx_reduction = 0.f, cohesion_sum = 0.f;
    // compute conflict focus sizes.
    unsigned int iters = 0;
    unsigned int idx;
    time_start = omp_get_wtime();
    for (i = 0; i < n; ++i){
        for (j = i + 1; j < n; ++j){
            // conflict_matrix[j + i * n] = 2.;
            conflict_matrix_int[j + i * n] = 2;
        }
    }
    conflict_loop_time += omp_get_wtime() - time_start;
    // if(print_out)
    //     print_matrix(n, n, conflict_matrix);


    //TODO: Add another level of blocking.
    for(xb = 0; xb < n; xb += block_size){
        x_block = ((xb + block_size) < n) ? block_size : (n - xb);
        for(yb = xb; yb < n; yb += block_size){
            y_block = ((yb + block_size) < n) ? block_size : (n - yb);
            time_start = omp_get_wtime();
            for (i = 0; i < x_block; ++i){
                memcpy(distance_xy_block + i * block_size, D + yb + (xb + i) * n, sizeof(float)*y_block);
            }

            // memset(buffer_conflict_xy_block, 0, sizeof(float)*block_size*block_size);
            memset(buffer_conflict_xy_block_int, 0, sizeof(int)*block_size*block_size);
            memops_loop_time += omp_get_wtime() - time_start;

            // copy DXY block from D.
            for(zb = yb; zb < n; zb += block_size){
                z_block = ((zb + block_size) < n) ? block_size : (n - zb);
                //copy DXZ and DYZ blocks from D.
                time_start = omp_get_wtime();

                for (i = 0; i < x_block; ++i){
                    memcpy(distance_xz_block + i * block_size, D + zb + (xb + i) * n, sizeof(float)*z_block);
                }
                for(i = 0; i < y_block; ++i){
                    memcpy(distance_yz_block + i * block_size, D + zb + (yb + i) * n, sizeof(float)*z_block);
                }

                // memset(buffer_conflict_xz_block, 0, sizeof(float)*block_size*block_size);
                // memset(buffer_conflict_yz_block, 0, sizeof(float)*block_size*block_size);
                memset(buffer_conflict_xz_block_int, 0, sizeof(int)*block_size*block_size);
                memset(buffer_conflict_yz_block_int, 0, sizeof(int)*block_size*block_size);
                memops_loop_time += omp_get_wtime() - time_start;

                time_start = omp_get_wtime();

                xend = (xb == yb && yb == zb) ? x_block - 1 : x_block;
                // ystart = 0;
                // zstart = 0;
                // if(xb == yb && yb == zb){
                //     xend = block_size - 1;
                // }
                for(x = 0; x < xend; ++x){
                    // if(xb == yb){
                    //     ystart = x + 1;
                    //     // conflict_yz_block += ystart*n;
                    // }
                    ystart = (xb == yb) ? x + 1 : 0;
                    // if(xb == yb){
                    //     for(y = x + 1; y < y_block; ++y){
                    //         // if(yb == zb){
                    //         //     zstart = y + 1;
                    //         // }
                    //         // xy_reduction = 0.f;
                    //         xy_reduction_int = 0;
                    //         zstart = (yb == zb) ? y + 1 : 0;
                    //         dist_xy = distance_xy_block[y + x * block_size];
                    //         // contains_tie = 0.f;
                    //         loop_len = z_block - zstart;
                    //         if(yb == zb){
                    //             // for (z = y + 1; z < block_size; ++z){
                    //             idx = y + 1;
                    //             #pragma unroll(16)
                    //             for (z = 0; z < loop_len; ++z){
                    //                 //compute masks for conflict blocks.

                    //                 distance_check_1_mask = dist_xy < distance_xz_block[idx + x * block_size];
                    //                 distance_check_2_mask = dist_xy < distance_yz_block[idx + y * block_size];
                    //                 scalar_xy_closest_int = distance_check_1_mask & distance_check_2_mask;

                    //                 distance_check_1_mask = distance_xz_block[idx + x * block_size] < dist_xy;
                    //                 distance_check_2_mask =  distance_xz_block[idx + x * block_size] < distance_yz_block[idx + y * block_size];
                    //                 scalar_xz_closest_int = distance_check_1_mask & distance_check_2_mask;

                    //                 distance_check_1_mask = distance_yz_block[idx + y * block_size] < distance_xz_block[idx + x * block_size];
                    //                 distance_check_2_mask = distance_yz_block[idx + y * block_size] < dist_xy;
                    //                 scalar_yz_closest_int = distance_check_1_mask & distance_check_2_mask;


                    //                 xy_reduction_int += scalar_xz_closest_int + scalar_yz_closest_int;
                    //                 buffer_conflict_yz_block_int[idx + y * block_size] += scalar_xy_closest_int + scalar_xz_closest_int;
                    //                 buffer_conflict_xz_block_int[idx + x * block_size] += scalar_xy_closest_int + scalar_yz_closest_int;

                    //                 idx++;
                    //             }

                    //             buffer_conflict_xy_block_int[y + x * block_size] += xy_reduction_int;
                    //         }
                    //         else{
                    //             // _mm512_store_epi32(buffer_conflict_xy_block_int + y + x * block_size, conflict_xy);
                    //             // __m512 dist_xy_avx = _mm512_set1_ps(dist_xy);
                    //             // __m512i all_ones = _mm512_set1_epi32(1);
                    //             // __m512 dist_xz_avx, dist_yz_avx;
                    //             // __mmask16 cmp_result_1, cmp_result_2, cmp_result_3;
                    //             // __m512i conf_xy, conf_xz, conf_yz;
                    //             // // for(z = 0; z < block_size; z+=16){
                    //             // //     dist_xz_avx = _mm512_load_ps(distance_xz_block + z + x * block_size);
                    //             // //     dist_yz_avx = _mm512_load_ps(distance_yz_block + z + y * block_size);
                    //             // //     distance_check_1_mask = _mm512_cmplt_ps_mask(dist_xy_avx, dist_xz_avx);
                    //             // //     distance_check_1_mask = _mm512_cmplt_ps_mask(dist_xy_avx, dist_yz_avx);
                    //             // //     cmp_result_1 = distance_check_1_mask & distance_check_2_mask;

                    //             // //     distance_check_1_mask = _mm512_cmplt_ps_mask(dist_xz_avx, dist_xy_avx);
                    //             // //     distance_check_1_mask = _mm512_cmplt_ps_mask(dist_xz_avx, dist_yz_avx);
                    //             // //     cmp_result_2 = distance_check_1_mask & distance_check_2_mask;

                    //             // //     distance_check_1_mask = _mm512_cmplt_ps_mask(dist_yz_avx, dist_xy_avx);
                    //             // //     distance_check_1_mask = _mm512_cmplt_ps_mask(dist_yz_avx, dist_xz_avx);
                    //             // //     cmp_result_3 = distance_check_1_mask & distance_check_2_mask;

                    //             // // }
                    //             #pragma unroll(16)
                    //             for (z = 0; z < z_block; ++z){
                    //                 //compute masks for conflict blocks.
                    //                 distance_check_1_mask = dist_xy < distance_xz_block[z + x * block_size];
                    //                 distance_check_2_mask = dist_xy < distance_yz_block[z + y * block_size];
                    //                 scalar_xy_closest_int = distance_check_1_mask & distance_check_2_mask;

                    //                 distance_check_1_mask = distance_xz_block[z + x * block_size] < dist_xy;
                    //                 distance_check_2_mask =  distance_xz_block[z + x * block_size] < distance_yz_block[z + y * block_size];
                    //                 scalar_xz_closest_int = distance_check_1_mask & distance_check_2_mask;

                    //                 distance_check_1_mask = distance_yz_block[z + y * block_size] < distance_xz_block[z + x * block_size];
                    //                 distance_check_2_mask = distance_yz_block[z + y * block_size] < dist_xy;
                    //                 scalar_yz_closest_int = distance_check_1_mask & distance_check_2_mask;

                    //                 xy_reduction_int += scalar_xz_closest_int + scalar_yz_closest_int;
                    //                 buffer_conflict_yz_block_int[z + y * block_size] += scalar_xy_closest_int + scalar_xz_closest_int;
                    //                 buffer_conflict_xz_block_int[z + x * block_size] += scalar_xy_closest_int + scalar_yz_closest_int;
                    //             }
                    //             buffer_conflict_xy_block_int[y + x * block_size] += xy_reduction_int;
                    //         }
                    //         // conflict_yz_block += n;
                    //     }
                    // }
                    // else{
                        for(y = ystart; y < y_block; ++y){
                            // if(yb == zb){
                            //     zstart = y + 1;
                            // }
                            // xy_reduction = 0.f;
                            xy_reduction_int = 0;
                            zstart = (yb == zb) ? y + 1 : 0;
                            dist_xy = distance_xy_block[y + x * block_size];
                            // contains_tie = 0.f;
                            loop_len = z_block - zstart;
                            if(yb == zb){
                                idx = y + 1;
                                #pragma unroll(16)
                                for (z = 0; z < loop_len; ++z){
                                    //compute masks for conflict blocks.

                                    distance_check_1_mask = dist_xy < distance_xz_block[idx + x * block_size];
                                    distance_check_2_mask = dist_xy < distance_yz_block[idx + y * block_size];
                                    scalar_xy_closest_int = distance_check_1_mask & distance_check_2_mask;

                                    distance_check_1_mask = distance_xz_block[idx + x * block_size] < dist_xy;
                                    distance_check_2_mask =  distance_xz_block[idx + x * block_size] < distance_yz_block[idx + y * block_size];
                                    scalar_xz_closest_int = distance_check_1_mask & distance_check_2_mask;

                                    distance_check_1_mask = distance_yz_block[idx + y * block_size] < distance_xz_block[idx + x * block_size];
                                    distance_check_2_mask = distance_yz_block[idx + y * block_size] < dist_xy;
                                    scalar_yz_closest_int = distance_check_1_mask & distance_check_2_mask;


                                    xy_reduction_int += scalar_xz_closest_int + scalar_yz_closest_int;
                                    buffer_conflict_yz_block_int[idx + y * block_size] += scalar_xy_closest_int + scalar_xz_closest_int;
                                    buffer_conflict_xz_block_int[idx + x * block_size] += scalar_xy_closest_int + scalar_yz_closest_int;

                                    idx++;
                                }
                                // buffer_conflict_xy_block_int[y + x * block_size] += xy_reduction_int;
                            }
                            else{
                                #pragma unroll(16)
                                for (z = 0; z < z_block; ++z){
                                    //compute masks for conflict blocks.
                                    distance_check_1_mask = dist_xy < distance_xz_block[z + x * block_size];
                                    distance_check_2_mask = dist_xy < distance_yz_block[z + y * block_size];
                                    scalar_xy_closest_int = distance_check_1_mask & distance_check_2_mask;

                                    distance_check_1_mask = distance_xz_block[z + x * block_size] < dist_xy;
                                    distance_check_2_mask =  distance_xz_block[z + x * block_size] < distance_yz_block[z + y * block_size];
                                    scalar_xz_closest_int = distance_check_1_mask & distance_check_2_mask;

                                    distance_check_1_mask = distance_yz_block[z + y * block_size] < distance_xz_block[z + x * block_size];
                                    distance_check_2_mask = distance_yz_block[z + y * block_size] < dist_xy;
                                    scalar_yz_closest_int = distance_check_1_mask & distance_check_2_mask;

                                    xy_reduction_int += scalar_xz_closest_int + scalar_yz_closest_int;
                                    buffer_conflict_yz_block_int[z + y * block_size] += scalar_xy_closest_int + scalar_xz_closest_int;
                                    buffer_conflict_xz_block_int[z + x * block_size] += scalar_xy_closest_int + scalar_yz_closest_int;
                                }
                                // buffer_conflict_xy_block_int[y + x * block_size] += xy_reduction_int;
                            }
                            buffer_conflict_xy_block_int[y + x * block_size] += xy_reduction_int;
                            // conflict_yz_block += n;
                        }
                    // }

                }
                conflict_loop_time += omp_get_wtime() - time_start;
                time_start2 = omp_get_wtime();
                // conflict_xy_block = conflict_matrix + yb + xb * n;
                // conflict_xz_block = conflict_matrix + zb + xb * n;
                // conflict_yz_block = conflict_matrix + zb + yb * n;
                conflict_xz_block_int = conflict_matrix_int + zb + xb * n;
                conflict_yz_block_int = conflict_matrix_int + zb + yb * n;

                for(i = 0; i < x_block; ++i){
                    for(j = 0; j < z_block; ++j){
                        // conflict_xz_block[j + i * n] += buffer_conflict_xz_block[j + i * block_size];
                        // conflict_yz_block[j + i * n] += buffer_conflict_yz_block[j + i * block_size];
                        conflict_xz_block_int[j + i * n] += buffer_conflict_xz_block_int[j + i * block_size];
                        // conflict_yz_block_int[j + i * n] += buffer_conflict_yz_block_int[j + i * block_size];
                    }
                }
                for(i = 0; i < y_block; ++i){
                    for(j = 0; j < z_block; ++j){
                        // conflict_xz_block[j + i * n] += buffer_conflict_xz_block[j + i * block_size];
                        // conflict_yz_block[j + i * n] += buffer_conflict_yz_block[j + i * block_size];
                        // conflict_xz_block_int[j + i * n] += buffer_conflict_xz_block_int[j + i * block_size];
                        conflict_yz_block_int[j + i * n] += buffer_conflict_yz_block_int[j + i * block_size];
                    }

                }
                memops_loop_time += omp_get_wtime() - time_start2;
                // printf("(xb: %d, yb: %d, zb: %d)\n", xb, yb, zb);
                // print_matrix_int(block_size, block_size, buffer_conflict_yz_block_int);
                // printf("[\n");
                // conflict_xy_block_int = conflict_matrix_int + yb + xb * n;
                // for(i = 0; i < block_size; ++i){
                //     for(j = 0; j < block_size; ++j){
                //         printf("%d ", conflict_xy_block_int[j + i * n] + buffer_conflict_xy_block_int[j + i * block_size]);
                //     }
                //     printf(";\n");
                // }
                // printf("];\n");

            }
            time_start2 = omp_get_wtime();
            // conflict_xy_block = conflict_matrix + yb + xb * n;
            conflict_xy_block_int = conflict_matrix_int + yb + xb * n;
            for(i = 0; i < x_block; ++i){
                for(j = 0; j < y_block; ++j){
                    // conflict_xy_block[j + i * n] += buffer_conflict_xy_block[j + i * block_size];
                    conflict_xy_block_int[j + i * n] += buffer_conflict_xy_block_int[j + i * block_size];
                }
                // conflict_xy_block += n;
            }
            // print_matrix_int(n,n, conflict_matrix_int);
            // printf("(xb: %d, yb: %d)\n", xb, yb);
            // print_matrix_int(block_size, block_size, buffer_conflict_xy_block_int);
            memops_loop_time += omp_get_wtime() - time_start2;
        }
    }
    // print_matrix_int(n, n, conflict_matrix_int);
    // return;
    time_start = omp_get_wtime();
    for(i = 0; i < n * n; ++i){
        // conflict_matrix[i] = 1.f/conflict_matrix[i];
        conflict_matrix[i] = 1.f/conflict_matrix_int[i];
    }
    // print_matrix_int(n, n, conflict_matrix_int);
    conflict_loop_time += omp_get_wtime() - time_start;
    // return;
    // printf("\n\n");
        // initialize diagonal of C.
    float sum;
    time_start = omp_get_wtime();
    for (i = 0; i < n; ++i){
        sum = 0.f;
        for (j = 0; j < i; ++j){
            sum += conflict_matrix[i + j * n];
        }
        for (j = i + 1; j < n; ++j){
            sum += conflict_matrix[j + i * n];
        }
        C[i + i * n] = sum;
    }
    cohesion_loop_time += omp_get_wtime() - time_start;
    iters = 0;

    // print_matrix(n, n, conflict_matrix);
    time_start = omp_get_wtime();
    _mm_free(conflict_matrix_int);
    _mm_free(buffer_conflict_xy_block_int); _mm_free(buffer_conflict_xz_block_int); _mm_free(buffer_conflict_yz_block_int);
    _mm_free(distance_xy_block); _mm_free(distance_xz_block); _mm_free(distance_yz_block);
    float *conflict_xy_block, *conflict_xz_block, *conflict_yz_block;
    float *cohesion_xy_block ;
    float *cohesion_yx_block;
    float *cohesion_xz_block;
    float *cohesion_zx_block;
    float *cohesion_yz_block;
    float *cohesion_zy_block;
    block_size/=2;

    // float* restrict mask_xy_closest = (float *) _mm_malloc(block_size * sizeof(float), VECALIGN);
    // float* restrict mask_xz_closest = (float *) _mm_malloc(block_size * sizeof(float), VECALIGN);
    // float* restrict mask_yz_closest = (float *) _mm_malloc(block_size * sizeof(float), VECALIGN);

    float scalar_xy_closest, scalar_xz_closest, scalar_yz_closest;

    // float* restrict mask_tie_xy_xz = (float *) _mm_malloc(block_size * sizeof(float), VECALIGN);
    // float* restrict mask_tie_xy_yz = (float *) _mm_malloc(block_size * sizeof(float), VECALIGN);
    // float* restrict mask_tie_xz_yz = (float *) _mm_malloc(block_size * sizeof(float), VECALIGN);

    float* buffer_conflict_xz_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
    float* buffer_conflict_yz_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
    // float* buffer_conflict_xy_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);

    float* restrict buffer_zx_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
    float* restrict buffer_zy_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
    float* restrict buffer_xz_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
    float* restrict buffer_yz_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
    float* restrict buffer_xy_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
    float* restrict buffer_yx_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);

    distance_xy_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
    distance_xz_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
    distance_yz_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);

    float* tmp_buffer_conflict_yz_block = buffer_conflict_yz_block;
    float* tmp_distance_yz_block = distance_yz_block;
    float* tmp_buffer_yz_block = buffer_yz_block;
    float* tmp_buffer_zy_block = buffer_zy_block;
    memops_loop_time += omp_get_wtime() - time_start;

    for(xb = 0; xb < n; xb += block_size){
        x_block = ((xb + block_size) < n) ? block_size : (n - xb);
        for(yb = xb; yb < n; yb += block_size){
            y_block = ((yb + block_size) < n) ? block_size : (n - yb);
            time_start = omp_get_wtime();
            for (i = 0; i < x_block; ++i){
                memcpy(distance_xy_block + i * block_size, D + yb + (xb + i) * n, sizeof(float)*y_block);
            }
            conflict_xy_block = conflict_matrix + yb + xb * n;
            memset(buffer_yx_block,0,sizeof(float)*block_size*block_size);
            memset(buffer_xy_block,0,sizeof(float)*block_size*block_size);
            memops_loop_time += omp_get_wtime() - time_start;
            for(zb = yb; zb < n; zb += block_size){
                z_block = ((zb + block_size) < n) ? block_size : (n - zb);
                time_start = omp_get_wtime();
                conflict_xz_block = conflict_matrix + zb + xb * n;
                conflict_yz_block = conflict_matrix + zb + yb * n;
                if(xb == yb){
                    #pragma unroll(8)
                    for (i = 0; i < x_block; ++i){
                        memcpy(distance_xz_block + i * block_size, D + zb + (xb + i) * n, sizeof(float)*z_block);
                        memcpy(buffer_conflict_xz_block + i * block_size, conflict_xz_block + i * n, sizeof(float)*z_block);
                        // memcpy(distance_yz_block + i * block_size, D + zb + (yb + i) * n, sizeof(float)*block_size);
                        // memcpy(buffer_conflict_yz_block + i * block_size, conflict_yz_block + i * n, sizeof(float)*block_size);
                    }
                    distance_yz_block = distance_xz_block;
                    buffer_conflict_yz_block = buffer_conflict_xz_block;
                    memset(buffer_zx_block,0,sizeof(float)*block_size*block_size);
                    memset(buffer_xz_block,0,sizeof(float)*block_size*block_size);

                    buffer_zy_block = buffer_zx_block;
                    buffer_yz_block = buffer_xz_block;
                }
                else{
                    distance_yz_block = tmp_distance_yz_block;
                    buffer_conflict_yz_block = tmp_buffer_conflict_yz_block;
                    buffer_zy_block = tmp_buffer_zy_block;
                    buffer_yz_block = tmp_buffer_yz_block;
                    #pragma unroll(8)
                    for (i = 0; i < x_block; ++i){
                        memcpy(distance_xz_block + i * block_size, D + zb + (xb + i) * n, sizeof(float)*z_block);
                        memcpy(buffer_conflict_xz_block + i * block_size, conflict_xz_block + i * n, sizeof(float)*z_block);
                        // memcpy(distance_yz_block + i * block_size, D + zb + (yb + i) * n, sizeof(float)*block_size);
                        // memcpy(buffer_conflict_yz_block + i * block_size, conflict_yz_block + i * n, sizeof(float)*block_size);
                    }
                    #pragma unroll(8)
                    for (i = 0; i < y_block; ++i){
                        // memcpy(distance_xz_block + i * block_size, D + zb + (xb + i) * n, sizeof(float)*block_size);
                        // memcpy(buffer_conflict_xz_block + i * block_size, conflict_xz_block + i * n, sizeof(float)*block_size);
                        memcpy(distance_yz_block + i * block_size, D + zb + (yb + i) * n, sizeof(float)*z_block);
                        memcpy(buffer_conflict_yz_block + i * block_size, conflict_yz_block + i * n, sizeof(float)*z_block);
                    }

                    memset(buffer_zx_block,0,sizeof(float)*block_size*block_size);
                    memset(buffer_zy_block,0,sizeof(float)*block_size*block_size);
                    memset(buffer_xz_block,0,sizeof(float)*block_size*block_size);
                    memset(buffer_yz_block,0,sizeof(float)*block_size*block_size);
                }
                memops_loop_time += omp_get_wtime() - time_start;

                time_start = omp_get_wtime();
                xend = x_block;
                // ystart = 0;
                // zstart = 0;
                if(xb == yb && yb == zb){
                    xend = x_block - 1;
                }
                for(x = 0; x < xend; ++x){
                    // if(xb == yb){
                    //     ystart = x + 1;

                    // }
                    ystart = (xb == yb) ? x + 1 : 0;
                    for(y = ystart; y < y_block; ++y){
                        // if(yb == zb){
                        //     zstart = y + 1;
                        // }
                        // zstart = (yb == zb) ? y + 1 : 0;
                        dist_xy = distance_xy_block[y + x * block_size];
                        // loop_len = block_size - zstart;
                        if(yb == zb){
                            loop_len = z_block - y - 1;
                            conflict_xy_val = conflict_xy_block[y + x * n];
                            xy_reduction = 0.f; yx_reduction = 0.f;
                            // for (z = y + 1; z < block_size; ++z){
                            for (z = 0; z < loop_len; ++z){
                                //compute masks for conflict blocks.
                                idx = z + y + 1;
                                distance_check_1_mask = dist_xy < distance_xz_block[idx + x * block_size];
                                distance_check_2_mask = dist_xy < distance_yz_block[idx + y * block_size];
                                scalar_xy_closest = distance_check_1_mask & distance_check_2_mask;

                                distance_check_1_mask = distance_xz_block[idx + x * block_size] < dist_xy;
                                distance_check_2_mask =  distance_xz_block[idx + x * block_size] < distance_yz_block[idx + y * block_size];
                                scalar_xz_closest = distance_check_1_mask & distance_check_2_mask;

                                distance_check_1_mask = distance_yz_block[idx + y * block_size] < distance_xz_block[idx + x * block_size];
                                distance_check_2_mask = distance_yz_block[idx + y * block_size] < dist_xy;
                                scalar_yz_closest = distance_check_1_mask & distance_check_2_mask;

                                // xy closest pair.
                                yx_reduction += scalar_xy_closest*buffer_conflict_xz_block[idx + x * block_size];
                                xy_reduction += scalar_xy_closest*buffer_conflict_yz_block[idx + y * block_size];

                                // xz closest pair.
                                buffer_xz_block[idx + x * block_size] += scalar_xz_closest*buffer_conflict_yz_block[idx + y * block_size];
                                buffer_zx_block[idx + x * block_size] += scalar_xz_closest*conflict_xy_val;

                                // yz closest pair.
                                buffer_yz_block[idx + y * block_size] += scalar_yz_closest*buffer_conflict_xz_block[idx + x * block_size];
                                buffer_zy_block[idx + y * block_size] += scalar_yz_closest*conflict_xy_val;
                            }
                            buffer_xy_block[y + x * block_size] += xy_reduction;
                            buffer_yx_block[y + x * block_size] += yx_reduction;
                        }
                        else{
                            conflict_xy_val = conflict_xy_block[y + x*n];
                            xy_reduction = 0.f; yx_reduction = 0.f;
                            // __m512 all_ones = _mm512_set1_ps(1.f);
                            // __m512 dist_xy_avx = _mm512_set1_ps(dist_xy);
                            // __m512 conflict_xy_avx = _mm512_set1_ps(conflict_xy_val);

                            // __m512 xy_reduction_avx = _mm512_setzero();
                            // __m512 yx_reduction_avx = _mm512_setzero();
                            // __m512 dist_xz_avx, dist_yz_avx, conflict_xz_avx, conflict_yz_avx;
                            // __m512 cohesion_xz_avx, cohesion_zx_avx, cohesion_yz_avx, cohesion_zy_avx;
                            // __mmask16 mask_xy_closest, mask_xz_closest, mask_yz_closest;

                            // __m512 xy_reduction_avx_1 = _mm512_setzero();
                            // __m512 yx_reduction_avx_1 = _mm512_setzero();
                            // __m512 dist_xz_avx_1, dist_yz_avx_1, conflict_xz_avx_1, conflict_yz_avx_1;
                            // __m512 cohesion_xz_avx_1, cohesion_zx_avx_1, cohesion_yz_avx_1, cohesion_zy_avx_1;
                            // __mmask16 mask_xy_closest_1, mask_xz_closest_1, mask_yz_closest_1;

                            // __m512 xy_reduction_avx_2 = _mm512_setzero();
                            // __m512 yx_reduction_avx_2 = _mm512_setzero();
                            // __m512 dist_xz_avx_2, dist_yz_avx_2, conflict_xz_avx_2, conflict_yz_avx_2;
                            // __m512 cohesion_xz_avx_2, cohesion_zx_avx_2, cohesion_yz_avx_2, cohesion_zy_avx_2;
                            // __mmask16 mask_xy_closest_2, mask_xz_closest_2, mask_yz_closest_2;

                            // __m512 xy_reduction_avx_3 = _mm512_setzero();
                            // __m512 yx_reduction_avx_3 = _mm512_setzero();
                            // __m512 dist_xz_avx_3, dist_yz_avx_3, conflict_xz_avx_3, conflict_yz_avx_3;
                            // __m512 cohesion_xz_avx_3, cohesion_zx_avx_3, cohesion_yz_avx_3, cohesion_zy_avx_3;
                            // __mmask16 mask_xy_closest_3, mask_xz_closest_3, mask_yz_closest_3;

                            // __mmask16 distance_check_3_mask, distance_check_4_mask;
                            // __mmask16 distance_check_5_mask, distance_check_6_mask;
                            // __mmask16 distance_check_7_mask, distance_check_8_mask;

                            // for(z = 0; z < block_size; z += 128){
                            //     dist_xz_avx = _mm512_load_ps(distance_xz_block + z + x * block_size);
                            //     dist_yz_avx = _mm512_load_ps(distance_yz_block + z + y * block_size);
                            //     cohesion_xz_avx = _mm512_load_ps(buffer_xz_block + z + x * block_size);
                            //     cohesion_zx_avx = _mm512_load_ps(buffer_zx_block + z + x * block_size);
                            //     cohesion_yz_avx = _mm512_load_ps(buffer_yz_block + z + y * block_size);
                            //     cohesion_zy_avx = _mm512_load_ps(buffer_zy_block + z + y * block_size);
                            //     conflict_yz_avx = _mm512_load_ps(buffer_conflict_yz_block + z + y * block_size);
                            //     conflict_xz_avx = _mm512_load_ps(buffer_conflict_xz_block + z + x * block_size);

                            //     dist_xz_avx_1 = _mm512_load_ps(distance_xz_block + z + 16 + x * block_size);
                            //     dist_yz_avx_1 = _mm512_load_ps(distance_yz_block + z + 16 + y * block_size);
                            //     cohesion_xz_avx_1 = _mm512_load_ps(buffer_xz_block + z + 16 + x * block_size);
                            //     cohesion_zx_avx_1 = _mm512_load_ps(buffer_zx_block + z + 16 + x * block_size);
                            //     cohesion_yz_avx_1 = _mm512_load_ps(buffer_yz_block + z + 16 + y * block_size);
                            //     cohesion_zy_avx_1 = _mm512_load_ps(buffer_zy_block + z + 16 + y * block_size);
                            //     conflict_yz_avx_1 = _mm512_load_ps(buffer_conflict_yz_block + z + 16 + y * block_size);
                            //     conflict_xz_avx_1 = _mm512_load_ps(buffer_conflict_xz_block + z + 16 + x * block_size);

                            //     dist_xz_avx_2 = _mm512_load_ps(distance_xz_block + z + 32 + x * block_size);
                            //     dist_yz_avx_2 = _mm512_load_ps(distance_yz_block + z + 32 + y * block_size);
                            //     cohesion_xz_avx_2 = _mm512_load_ps(buffer_xz_block + z + 32 + x * block_size);
                            //     cohesion_zx_avx_2 = _mm512_load_ps(buffer_zx_block + z + 32 + x * block_size);
                            //     cohesion_yz_avx_2 = _mm512_load_ps(buffer_yz_block + z + 32 + y * block_size);
                            //     cohesion_zy_avx_2 = _mm512_load_ps(buffer_zy_block + z + 32 + y * block_size);
                            //     conflict_yz_avx_2 = _mm512_load_ps(buffer_conflict_yz_block + z + 32 + y * block_size);
                            //     conflict_xz_avx_2 = _mm512_load_ps(buffer_conflict_xz_block + z + 32 + x * block_size);

                            //     dist_xz_avx_3 = _mm512_load_ps(distance_xz_block + z + 64 + x * block_size);
                            //     dist_yz_avx_3 = _mm512_load_ps(distance_yz_block + z + 64 + y * block_size);
                            //     cohesion_xz_avx_3 = _mm512_load_ps(buffer_xz_block + z + 64 + x * block_size);
                            //     cohesion_zx_avx_3 = _mm512_load_ps(buffer_zx_block + z + 64 + x * block_size);
                            //     cohesion_yz_avx_3 = _mm512_load_ps(buffer_yz_block + z + 64 + y * block_size);
                            //     cohesion_zy_avx_3 = _mm512_load_ps(buffer_zy_block + z + 64 + y * block_size);
                            //     conflict_yz_avx_3 = _mm512_load_ps(buffer_conflict_yz_block + z + 64 + y * block_size);
                            //     conflict_xz_avx_3 = _mm512_load_ps(buffer_conflict_xz_block + z + 64 + x * block_size);

                            //     distance_check_1_mask = _mm512_cmplt_ps_mask(dist_xy_avx, dist_xz_avx);
                            //     distance_check_2_mask = _mm512_cmplt_ps_mask(dist_xy_avx, dist_yz_avx);
                            //     mask_xy_closest = distance_check_1_mask & distance_check_2_mask;

                            //     distance_check_1_mask = _mm512_cmplt_ps_mask(dist_xz_avx, dist_xy_avx);
                            //     distance_check_2_mask = _mm512_cmplt_ps_mask(dist_xz_avx, dist_yz_avx);
                            //     mask_xz_closest = distance_check_1_mask & distance_check_2_mask;

                            //     distance_check_1_mask = _mm512_cmplt_ps_mask(dist_yz_avx, dist_xz_avx);
                            //     distance_check_2_mask = _mm512_cmplt_ps_mask(dist_yz_avx, dist_xy_avx);
                            //     mask_yz_closest = distance_check_1_mask & distance_check_2_mask;

                            //     distance_check_3_mask = _mm512_cmplt_ps_mask(dist_xy_avx, dist_xz_avx);
                            //     distance_check_4_mask = _mm512_cmplt_ps_mask(dist_xy_avx, dist_yz_avx);
                            //     mask_xy_closest_1 = distance_check_3_mask & distance_check_4_mask;

                            //     distance_check_3_mask = _mm512_cmplt_ps_mask(dist_xz_avx, dist_xy_avx);
                            //     distance_check_4_mask = _mm512_cmplt_ps_mask(dist_xz_avx, dist_yz_avx);
                            //     mask_xz_closest_1 = distance_check_3_mask & distance_check_4_mask;

                            //     distance_check_3_mask = _mm512_cmplt_ps_mask(dist_yz_avx, dist_xz_avx);
                            //     distance_check_4_mask = _mm512_cmplt_ps_mask(dist_yz_avx, dist_xy_avx);
                            //     mask_yz_closest_1 = distance_check_3_mask & distance_check_4_mask;

                            //     distance_check_5_mask = _mm512_cmplt_ps_mask(dist_xy_avx, dist_xz_avx);
                            //     distance_check_6_mask = _mm512_cmplt_ps_mask(dist_xy_avx, dist_yz_avx);
                            //     mask_xy_closest_2 = distance_check_5_mask & distance_check_6_mask;

                            //     distance_check_5_mask = _mm512_cmplt_ps_mask(dist_xz_avx, dist_xy_avx);
                            //     distance_check_6_mask = _mm512_cmplt_ps_mask(dist_xz_avx, dist_yz_avx);
                            //     mask_xz_closest_2 = distance_check_5_mask & distance_check_6_mask;

                            //     distance_check_5_mask = _mm512_cmplt_ps_mask(dist_yz_avx, dist_xz_avx);
                            //     distance_check_6_mask = _mm512_cmplt_ps_mask(dist_yz_avx, dist_xy_avx);
                            //     mask_yz_closest_2 = distance_check_5_mask & distance_check_6_mask;

                            //     distance_check_7_mask = _mm512_cmplt_ps_mask(dist_xy_avx, dist_xz_avx);
                            //     distance_check_8_mask = _mm512_cmplt_ps_mask(dist_xy_avx, dist_yz_avx);
                            //     mask_xy_closest_3 = distance_check_7_mask & distance_check_8_mask;

                            //     distance_check_7_mask = _mm512_cmplt_ps_mask(dist_xz_avx, dist_xy_avx);
                            //     distance_check_8_mask = _mm512_cmplt_ps_mask(dist_xz_avx, dist_yz_avx);
                            //     mask_xz_closest_3 = distance_check_7_mask & distance_check_8_mask;

                            //     distance_check_7_mask = _mm512_cmplt_ps_mask(dist_yz_avx, dist_xz_avx);
                            //     distance_check_8_mask = _mm512_cmplt_ps_mask(dist_yz_avx, dist_xy_avx);
                            //     mask_yz_closest_3 = distance_check_7_mask & distance_check_8_mask;


                            //     xy_reduction_avx = _mm512_mask_fmadd_ps(all_ones, mask_xy_closest, conflict_yz_avx, xy_reduction_avx);
                            //     yx_reduction_avx = _mm512_mask_fmadd_ps(all_ones, mask_xy_closest, conflict_xz_avx, yx_reduction_avx);

                            //     xy_reduction_avx_1 = _mm512_mask_fmadd_ps(all_ones, mask_xy_closest_1, conflict_yz_avx_1, xy_reduction_avx_1);
                            //     yx_reduction_avx_1 = _mm512_mask_fmadd_ps(all_ones, mask_xy_closest_1, conflict_xz_avx_1, yx_reduction_avx_1);

                            //     xy_reduction_avx_2 = _mm512_mask_fmadd_ps(all_ones, mask_xy_closest_2, conflict_yz_avx_2, xy_reduction_avx_2);
                            //     yx_reduction_avx_2 = _mm512_mask_fmadd_ps(all_ones, mask_xy_closest_2, conflict_xz_avx_2, yx_reduction_avx_2);

                            //     xy_reduction_avx_3 = _mm512_mask_fmadd_ps(all_ones, mask_xy_closest_3, conflict_yz_avx_3, xy_reduction_avx_3);
                            //     yx_reduction_avx_3 = _mm512_mask_fmadd_ps(all_ones, mask_xy_closest_3, conflict_xz_avx_3, yx_reduction_avx_3);

                            //     _mm512_store_ps(buffer_xz_block + z + x * block_size, _mm512_mask_fmadd_ps(all_ones, mask_xz_closest, conflict_yz_avx, cohesion_xz_avx));
                            //     _mm512_store_ps(buffer_zx_block + z + x * block_size, _mm512_mask_fmadd_ps(all_ones, mask_xz_closest, conflict_xy_avx, cohesion_zx_avx));
                            //     _mm512_store_ps(buffer_yz_block + z + y * block_size, _mm512_mask_fmadd_ps(all_ones, mask_yz_closest, conflict_xz_avx, cohesion_yz_avx));
                            //     _mm512_store_ps(buffer_zy_block + z + y * block_size, _mm512_mask_fmadd_ps(all_ones, mask_yz_closest, conflict_xy_avx, cohesion_zy_avx));

                            //     _mm512_store_ps(buffer_xz_block + z + 16 + x * block_size, _mm512_mask_fmadd_ps(all_ones, mask_xz_closest_1, conflict_yz_avx_1, cohesion_xz_avx_1));
                            //     _mm512_store_ps(buffer_zx_block + z + 16 + x * block_size, _mm512_mask_fmadd_ps(all_ones, mask_xz_closest_1, conflict_xy_avx, cohesion_zx_avx_1));
                            //     _mm512_store_ps(buffer_yz_block + z + 16 + y * block_size, _mm512_mask_fmadd_ps(all_ones, mask_yz_closest_1, conflict_xz_avx_1, cohesion_yz_avx_1));
                            //     _mm512_store_ps(buffer_zy_block + z + 16 + y * block_size, _mm512_mask_fmadd_ps(all_ones, mask_yz_closest_1, conflict_xy_avx, cohesion_zy_avx_1));

                            //     _mm512_store_ps(buffer_xz_block + z + 32 + x * block_size, _mm512_mask_fmadd_ps(all_ones, mask_xz_closest_2, conflict_yz_avx_2, cohesion_xz_avx_2));
                            //     _mm512_store_ps(buffer_zx_block + z + 32 + x * block_size, _mm512_mask_fmadd_ps(all_ones, mask_xz_closest_2, conflict_xy_avx, cohesion_zx_avx_2));
                            //     _mm512_store_ps(buffer_yz_block + z + 32 + y * block_size, _mm512_mask_fmadd_ps(all_ones, mask_yz_closest_2, conflict_xz_avx_2, cohesion_yz_avx_2));
                            //     _mm512_store_ps(buffer_zy_block + z + 32 + y * block_size, _mm512_mask_fmadd_ps(all_ones, mask_yz_closest_2, conflict_xy_avx, cohesion_zy_avx_2));

                            //     _mm512_store_ps(buffer_xz_block + z + 64 + x * block_size, _mm512_mask_fmadd_ps(all_ones, mask_xz_closest_3, conflict_yz_avx_3, cohesion_xz_avx_3));
                            //     _mm512_store_ps(buffer_zx_block + z + 64 + x * block_size, _mm512_mask_fmadd_ps(all_ones, mask_xz_closest_3, conflict_xy_avx, cohesion_zx_avx_3));
                            //     _mm512_store_ps(buffer_yz_block + z + 64 + y * block_size, _mm512_mask_fmadd_ps(all_ones, mask_yz_closest_3, conflict_xz_avx_3, cohesion_yz_avx_3));
                            //     _mm512_store_ps(buffer_zy_block + z + 64 + y * block_size, _mm512_mask_fmadd_ps(all_ones, mask_yz_closest_3, conflict_xy_avx, cohesion_zy_avx_3));
                            // }
                            // xy_reduction_avx += xy_reduction_avx_1 + xy_reduction_avx_2 + xy_reduction_avx_3;
                            // yx_reduction_avx += yx_reduction_avx_1 + yx_reduction_avx_2 + yx_reduction_avx_3;
                            // xy_reduction += _mm512_reduce_add_ps(xy_reduction_avx);
                            // yx_reduction += _mm512_reduce_add_ps(yx_reduction_avx);
                            #pragma unroll(8)
                            //update cohesion blocks.
                            for (z = 0; z < z_block; ++z){
                                distance_check_1_mask = dist_xy < distance_xz_block[z + x * block_size];
                                distance_check_2_mask = dist_xy < distance_yz_block[z + y * block_size];
                                scalar_xy_closest = distance_check_1_mask & distance_check_2_mask;

                                distance_check_1_mask = distance_xz_block[z + x * block_size] < dist_xy;
                                distance_check_2_mask =  distance_xz_block[z + x * block_size] < distance_yz_block[z + y * block_size];
                                scalar_xz_closest = distance_check_1_mask & distance_check_2_mask;

                                distance_check_1_mask = distance_yz_block[z + y * block_size] < distance_xz_block[z + x * block_size];
                                distance_check_2_mask = distance_yz_block[z + y * block_size] < dist_xy;
                                scalar_yz_closest = distance_check_1_mask & distance_check_2_mask;

                                // xy closest pair.
                                xy_reduction += scalar_xy_closest*buffer_conflict_yz_block[z + y * block_size];
                                yx_reduction += scalar_xy_closest*buffer_conflict_xz_block[z + x * block_size];

                                // xz closest pair
                                buffer_xz_block[z + x * block_size] += scalar_xz_closest*buffer_conflict_yz_block[z + y * block_size];
                                buffer_zx_block[z + x * block_size] += scalar_xz_closest*conflict_xy_val;

                                //yz closest pair.
                                buffer_yz_block[z + y * block_size] += scalar_yz_closest*buffer_conflict_xz_block[z + x * block_size];
                                buffer_zy_block[z + y * block_size] += scalar_yz_closest*conflict_xy_val;
                            }
                            buffer_xy_block[y + x * block_size] += xy_reduction;
                            buffer_yx_block[y + x * block_size] += yx_reduction;
                        }
                    }
                    // conflict_xy_block += n;
                }
                cohesion_loop_time += omp_get_wtime() - time_start;
                time_start2 = omp_get_wtime();
                if(xb == yb){
                    cohesion_zx_block = C + xb + zb * n;
                    cohesion_xz_block = C + zb + xb * n;
                    for(i = 0; i < z_block; ++i){
                        for(j = 0; j < x_block; ++j){
                            // printf("idx: %d\n", n*j + i);
                            cohesion_zx_block[j] += buffer_zx_block[i + j * block_size];
                            // cohesion_yx_block[j] += buffer_yx_block[i + j * block_size];
                        }
                        cohesion_zx_block += n;
                    }
                    for(i = 0; i < x_block; ++i){
                        for(j = 0; j < z_block; ++j){
                            cohesion_xz_block[j] += buffer_xz_block[j + i * block_size];
                            // cohesion_xy_block[j] += buffer_xy_block[j + i * block_size];
                        }
                        cohesion_xz_block += n;
                    }

                }
                else{
                    cohesion_zx_block = C + xb + zb * n;
                    cohesion_zy_block = C + yb + zb * n;
                    cohesion_xz_block = C + zb + xb * n;
                    cohesion_yz_block = C + zb + yb * n;
                    for(i = 0; i < z_block; ++i){
                        for(j = 0; j < x_block; ++j){
                            // printf("idx: %d\n", n*j + i);
                            cohesion_zx_block[j] += buffer_zx_block[i + j * block_size];
                            // cohesion_zy_block[j] += buffer_zy_block[i + j * block_size];
                            // cohesion_yx_block[j] += buffer_yx_block[i + j * block_size];
                        }
                        cohesion_zx_block += n;
                        // cohesion_zy_block += n;
                    }
                    for(i = 0; i < z_block; ++i){
                        for(j = 0; j < y_block; ++j){
                            // printf("idx: %d\n", n*j + i);
                            // cohesion_zx_block[j] += buffer_zx_block[i + j * block_size];
                            cohesion_zy_block[j] += buffer_zy_block[i + j * block_size];
                            // cohesion_yx_block[j] += buffer_yx_block[i + j * block_size];
                        }
                        // cohesion_zx_block += n;
                        cohesion_zy_block += n;
                    }
                    for(i = 0; i < x_block; ++i){
                        for(j = 0; j < z_block; ++j){
                            cohesion_xz_block[j] += buffer_xz_block[j + i * block_size];
                            // cohesion_yz_block[j] += buffer_yz_block[j + i * block_size];
                            // cohesion_xy_block[j] += buffer_xy_block[j + i * block_size];
                        }
                        cohesion_xz_block += n;
                        // cohesion_yz_block += n;
                    }
                    for(i = 0; i < y_block; ++i){
                        for(j = 0; j < z_block; ++j){
                            // cohesion_xz_block[j] += buffer_xz_block[j + i * block_size];
                            cohesion_yz_block[j] += buffer_yz_block[j + i * block_size];
                            // cohesion_xy_block[j] += buffer_xy_block[j + i * block_size];
                        }
                        // cohesion_xz_block += n;
                        cohesion_yz_block += n;
                    }
                }

                memops_loop_time += omp_get_wtime() - time_start2;
            }
            time_start2 = omp_get_wtime();
            cohesion_xy_block = C + yb + xb * n;
            cohesion_yx_block = C + xb + yb * n;
            for(i = 0; i < y_block; ++i){
                for(j = 0; j < x_block; ++j){
                    // printf("idx: %d\n", n*j + i);
                    cohesion_yx_block[j] += buffer_yx_block[i + j * block_size];
                }
                cohesion_yx_block += n;
            }

            for(i = 0; i < x_block; ++i){
                for(j = 0; j < y_block; ++j){
                    cohesion_xy_block[j] += buffer_xy_block[j + i * block_size];
                }
                cohesion_xy_block += n;
            }
            memops_loop_time += omp_get_wtime() - time_start2;
        }
    }
    // print_matrix(n, n, C);

    printf("==============================================\n");
    printf("Seq. Triplet Explict Intrinsics Loop Times\n");
    printf("==============================================\n");

    printf("memops loop time: %.5fs\n", memops_loop_time);
    printf("conflict focus size loop time: %.5fs\n", conflict_loop_time);
    printf("cohesion matrix update loop time: %.5fs\n\n", cohesion_loop_time);

    _mm_free(distance_xy_block); _mm_free(distance_xz_block); _mm_free(tmp_distance_yz_block);
    // _mm_free(mask_tie_xy_xz); _mm_free(mask_tie_xy_yz); _mm_free(mask_tie_xz_yz);
    _mm_free(buffer_zx_block); _mm_free(tmp_buffer_zy_block); _mm_free(buffer_yx_block);
    _mm_free(buffer_xz_block); _mm_free(tmp_buffer_yz_block); _mm_free(buffer_xy_block);
    // _mm_free(mask_xy_closest); _mm_free(mask_xz_closest); _mm_free(mask_yz_closest);
    _mm_free(buffer_conflict_xz_block); _mm_free(tmp_buffer_conflict_yz_block);
    // _mm_free(buffer_conflict_xy_block);
    _mm_free(conflict_matrix);
    // _mm_free(conflict_matrix_int);
    // _mm_free(mask_xy_closest_int); _mm_free(mask_xz_closest_int); _mm_free(mask_yz_closest_int);
    // _mm_free(buffer_conflict_xz_block_int); _mm_free(buffer_conflict_yz_block_int); _mm_free(buffer_conflict_xy_block_int);

}



void pald_triplet_fewercompares(float *D, float beta, int n, float *C, int block_size){
    //TODO: Optimized sequential triplet code.
    float* restrict conflict_matrix = (float *)  _mm_malloc(n * n * sizeof(float), VECALIGN);
    // memset(conflict_matrix, 0, n * n * sizeof(float));
    unsigned int* restrict conflict_matrix_int = (unsigned int*)  _mm_malloc(n * n * sizeof(unsigned int), VECALIGN);
    // memset(conflict_matrix_int, 0, n * n * sizeof(unsigned int));

    float* restrict distance_xy_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
    float* restrict distance_xz_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
    float* restrict distance_yz_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);

    // float* restrict mask_xy_closest = (float *) _mm_malloc(block_size * sizeof(float), VECALIGN);
    // float* restrict mask_xz_closest = (float *) _mm_malloc(block_size * sizeof(float), VECALIGN);
    // float* restrict mask_yz_closest = (float *) _mm_malloc(block_size * sizeof(float), VECALIGN);

    // unsigned int* restrict mask_xy_closest_int = (unsigned int*) _mm_malloc(block_size * sizeof(unsigned int), VECALIGN);
    // unsigned int* restrict mask_xz_closest_int = (unsigned int*) _mm_malloc(block_size * sizeof(unsigned int), VECALIGN);
    // unsigned int* restrict mask_yz_closest_int = (unsigned int*) _mm_malloc(block_size * sizeof(unsigned int), VECALIGN);

    unsigned int scalar_xy_closest_int, scalar_xz_closest_int, scalar_yz_closest_int;

    unsigned int* restrict buffer_conflict_xz_block_int = (unsigned int *) _mm_malloc(block_size * block_size * sizeof(unsigned int), VECALIGN);
    unsigned int* restrict buffer_conflict_yz_block_int = (unsigned int *) _mm_malloc(block_size * block_size * sizeof(unsigned int), VECALIGN);
    unsigned int* restrict buffer_conflict_xy_block_int = (unsigned int *) _mm_malloc(block_size * block_size * sizeof(unsigned int), VECALIGN);

    // float* restrict buffer_contains_tie = (float *) _mm_malloc(block_size * sizeof(float), VECALIGN);

    // char distance_check_1 = 0;
    // char distance_check_2 = 0;
    // char distance_check_3 = 0;
    unsigned int distance_check_1_mask, distance_check_2_mask, distance_check_3_mask;
    unsigned int xy_reduction_int;
    float dist_xy  = 0.f;
    float conflict_xy_val = 0.f;
    unsigned int loop_len = 0;

    unsigned int *conflict_xy_block_int, *conflict_xz_block_int, *conflict_yz_block_int;
    // char print_out = 0;
    double time_start = 0.0, time_start2 = 0.0;
    double memops_loop_time = 0.0, conflict_loop_time = 0.0, cohesion_loop_time = 0.0;
    unsigned int xb, yb, zb, x, y, z;
    unsigned int i, j, k;
    unsigned int x_block, y_block, z_block;
    // int size_xy = block_size, size_xz = block_size, size_yz = block_size;
    unsigned int xend, ystart, zstart;
    float xy_reduction = 0.f, yx_reduction = 0.f, cohesion_sum = 0.f;
    // compute conflict focus sizes.
    unsigned int iters = 0;
    unsigned int idx;
    time_start = omp_get_wtime();
    for (i = 0; i < n; ++i){
        for (j = i + 1; j < n; ++j){
            // conflict_matrix[j + i * n] = 2.;
            conflict_matrix_int[j + i * n] = 2;
        }
    }
    conflict_loop_time += omp_get_wtime() - time_start;
    // if(print_out)
    //     print_matrix(n, n, conflict_matrix);


    //TODO: Add another level of blocking.
    for(xb = 0; xb < n; xb += block_size){
        x_block = ((xb + block_size) < n) ? block_size : (n - xb);
        for(yb = xb; yb < n; yb += block_size){
            y_block = ((yb + block_size) < n) ? block_size : (n - yb);
            time_start = omp_get_wtime();
            for (i = 0; i < x_block; ++i){
                memcpy(distance_xy_block + i * block_size, D + yb + (xb + i) * n, sizeof(float)*y_block);
            }

            // memset(buffer_conflict_xy_block, 0, sizeof(float)*block_size*block_size);
            memset(buffer_conflict_xy_block_int, 0, sizeof(int)*block_size*block_size);
            memops_loop_time += omp_get_wtime() - time_start;

            // copy DXY block from D.
            for(zb = yb; zb < n; zb += block_size){
                z_block = ((zb + block_size) < n) ? block_size : (n - zb);
                //copy DXZ and DYZ blocks from D.
                time_start = omp_get_wtime();

                for (i = 0; i < x_block; ++i){
                    memcpy(distance_xz_block + i * block_size, D + zb + (xb + i) * n, sizeof(float)*z_block);
                }
                for(i = 0; i < y_block; ++i){
                    memcpy(distance_yz_block + i * block_size, D + zb + (yb + i) * n, sizeof(float)*z_block);
                }

                // memset(buffer_conflict_xz_block, 0, sizeof(float)*block_size*block_size);
                // memset(buffer_conflict_yz_block, 0, sizeof(float)*block_size*block_size);
                memset(buffer_conflict_xz_block_int, 0, sizeof(int)*block_size*block_size);
                memset(buffer_conflict_yz_block_int, 0, sizeof(int)*block_size*block_size);
                memops_loop_time += omp_get_wtime() - time_start;

                time_start = omp_get_wtime();

                xend = (xb == yb && yb == zb) ? x_block - 1 : x_block;
                // ystart = 0;
                // zstart = 0;
                // if(xb == yb && yb == zb){
                //     xend = block_size - 1;
                // }
                for(x = 0; x < xend; ++x){
                    // if(xb == yb){
                    //     ystart = x + 1;
                    //     // conflict_yz_block += ystart*n;
                    // }
                    ystart = (xb == yb) ? x + 1 : 0;
                    if(xb == yb){
                        for(y = x + 1; y < y_block; ++y){
                            // if(yb == zb){
                            //     zstart = y + 1;
                            // }
                            // xy_reduction = 0.f;
                            xy_reduction_int = 0;
                            zstart = (yb == zb) ? y + 1 : 0;
                            dist_xy = distance_xy_block[y + x * block_size];
                            // contains_tie = 0.f;
                            loop_len = z_block - zstart;
                            if(yb == zb){
                                // for (z = y + 1; z < block_size; ++z){
                                idx = y + 1;
                                #pragma unroll(16)
                                for (z = 0; z < loop_len; ++z){
                                    //compute masks for conflict blocks.

                                    distance_check_1_mask = (dist_xy < distance_xz_block[idx + x * block_size]);
                                    distance_check_2_mask = (dist_xy < distance_yz_block[idx + y * block_size]);
                                    distance_check_3_mask =  (distance_xz_block[idx + x * block_size] < distance_yz_block[idx + y * block_size]);

                                    scalar_xy_closest_int = distance_check_1_mask & distance_check_2_mask;

                                    // distance_check_1_mask = distance_xz_block[idx + x * block_size] < dist_xy;
                                    // distance_check_2_mask =  distance_xz_block[idx + x * block_size] < distance_yz_block[idx + y * block_size];
                                    scalar_xz_closest_int = (1 - distance_check_1_mask) & distance_check_3_mask;
                                    // distance_check_1_mask = distance_yz_block[idx + y * block_size] < distance_xz_block[idx + x * block_size];
                                    // distance_check_2_mask = distance_yz_block[idx + y * block_size] < dist_xy;
                                    scalar_yz_closest_int = (1 - distance_check_3_mask) & (1 - distance_check_2_mask);


                                    xy_reduction_int += scalar_xz_closest_int + scalar_yz_closest_int;
                                    buffer_conflict_yz_block_int[idx + y * block_size] += scalar_xy_closest_int + scalar_xz_closest_int;
                                    buffer_conflict_xz_block_int[idx + x * block_size] += scalar_xy_closest_int + scalar_yz_closest_int;

                                    idx++;
                                }

                                buffer_conflict_xy_block_int[y + x * block_size] += xy_reduction_int;
                            }
                            else{
                                // _mm512_store_epi32(buffer_conflict_xy_block_int + y + x * block_size, conflict_xy);
                                // __m512 dist_xy_avx = _mm512_set1_ps(dist_xy);
                                // __m512i all_ones = _mm512_set1_epi32(1);
                                // __m512 dist_xz_avx, dist_yz_avx;
                                // __mmask16 cmp_result_1, cmp_result_2, cmp_result_3;
                                // __m512i conf_xy, conf_xz, conf_yz;
                                // // for(z = 0; z < block_size; z+=16){
                                // //     dist_xz_avx = _mm512_load_ps(distance_xz_block + z + x * block_size);
                                // //     dist_yz_avx = _mm512_load_ps(distance_yz_block + z + y * block_size);
                                // //     distance_check_1_mask = _mm512_cmplt_ps_mask(dist_xy_avx, dist_xz_avx);
                                // //     distance_check_1_mask = _mm512_cmplt_ps_mask(dist_xy_avx, dist_yz_avx);
                                // //     cmp_result_1 = distance_check_1_mask & distance_check_2_mask;

                                // //     distance_check_1_mask = _mm512_cmplt_ps_mask(dist_xz_avx, dist_xy_avx);
                                // //     distance_check_1_mask = _mm512_cmplt_ps_mask(dist_xz_avx, dist_yz_avx);
                                // //     cmp_result_2 = distance_check_1_mask & distance_check_2_mask;

                                // //     distance_check_1_mask = _mm512_cmplt_ps_mask(dist_yz_avx, dist_xy_avx);
                                // //     distance_check_1_mask = _mm512_cmplt_ps_mask(dist_yz_avx, dist_xz_avx);
                                // //     cmp_result_3 = distance_check_1_mask & distance_check_2_mask;

                                // // }
                                #pragma unroll(16)
                                for (z = 0; z < z_block; ++z){
                                    //compute masks for conflict blocks.
                                    distance_check_1_mask = dist_xy < distance_xz_block[z + x * block_size];
                                    distance_check_2_mask = dist_xy < distance_yz_block[z + y * block_size];
                                    distance_check_3_mask =  distance_xz_block[z + x * block_size] < distance_yz_block[z + y * block_size];

                                    scalar_xy_closest_int = distance_check_1_mask & distance_check_2_mask;

                                    // distance_check_1_mask = distance_xz_block[z + x * block_size] < dist_xy;
                                    // distance_check_2_mask =  distance_xz_block[z + x * block_size] < distance_yz_block[z + y * block_size];
                                    scalar_xz_closest_int = 1 & (~distance_check_1_mask) & distance_check_3_mask;

                                    // distance_check_1_mask = distance_yz_block[z + y * block_size] < distance_xz_block[z + x * block_size];
                                    // distance_check_2_mask = distance_yz_block[z + y * block_size] < dist_xy;
                                    scalar_yz_closest_int = 1 & (~distance_check_3_mask) & (~distance_check_2_mask);

                                    xy_reduction_int += scalar_xz_closest_int + scalar_yz_closest_int;
                                    buffer_conflict_yz_block_int[z + y * block_size] += scalar_xy_closest_int + scalar_xz_closest_int;
                                    buffer_conflict_xz_block_int[z + x * block_size] += scalar_xy_closest_int + scalar_yz_closest_int;
                                }
                                buffer_conflict_xy_block_int[y + x * block_size] += xy_reduction_int;
                            }
                            // conflict_yz_block += n;
                        }
                    }
                    else{
                        for(y = 0; y < y_block; ++y){
                            // if(yb == zb){
                            //     zstart = y + 1;
                            // }
                            // xy_reduction = 0.f;
                            xy_reduction_int = 0;
                            zstart = (yb == zb) ? y + 1 : 0;
                            dist_xy = distance_xy_block[y + x * block_size];
                            // contains_tie = 0.f;
                            loop_len = z_block - zstart;
                            if(yb == zb){
                                idx = y + 1;
                                #pragma unroll(16)
                                for (z = 0; z < loop_len; ++z){
                                    //compute masks for conflict blocks.

                                    distance_check_1_mask = dist_xy < distance_xz_block[idx + x * block_size];
                                    distance_check_2_mask = dist_xy < distance_yz_block[idx + y * block_size];
                                    scalar_xy_closest_int = distance_check_1_mask & distance_check_2_mask;

                                    distance_check_1_mask = distance_xz_block[idx + x * block_size] < dist_xy;
                                    distance_check_2_mask =  distance_xz_block[idx + x * block_size] < distance_yz_block[idx + y * block_size];
                                    scalar_xz_closest_int = distance_check_1_mask & distance_check_2_mask;

                                    distance_check_1_mask = distance_yz_block[idx + y * block_size] < distance_xz_block[idx + x * block_size];
                                    distance_check_2_mask = distance_yz_block[idx + y * block_size] < dist_xy;
                                    scalar_yz_closest_int = distance_check_1_mask & distance_check_2_mask;


                                    xy_reduction_int += scalar_xz_closest_int + scalar_yz_closest_int;
                                    buffer_conflict_yz_block_int[idx + y * block_size] += scalar_xy_closest_int + scalar_xz_closest_int;
                                    buffer_conflict_xz_block_int[idx + x * block_size] += scalar_xy_closest_int + scalar_yz_closest_int;

                                    idx++;
                                }
                                buffer_conflict_xy_block_int[y + x * block_size] += xy_reduction_int;
                            }
                            else{
                                #pragma unroll(16)
                                for (z = 0; z < z_block; ++z){
                                    //compute masks for conflict blocks.
                                    distance_check_1_mask = (dist_xy < distance_xz_block[z + x * block_size]);
                                    distance_check_2_mask = (dist_xy < distance_yz_block[z + y * block_size]);
                                    distance_check_3_mask = (distance_xz_block[z + x * block_size] < distance_yz_block[z + y * block_size]) ;

                                    scalar_xy_closest_int = distance_check_1_mask & distance_check_2_mask;

                                    // distance_check_1_mask = distance_xz_block[z + x * block_size] < dist_xy;
                                    // distance_check_2_mask =  distance_xz_block[z + x * block_size] < distance_yz_block[z + y * block_size];
                                    scalar_xz_closest_int = (1 - distance_check_1_mask) & distance_check_3_mask;

                                    // distance_check_1_mask = distance_yz_block[z + y * block_size] < distance_xz_block[z + x * block_size];
                                    // distance_check_2_mask = distance_yz_block[z + y * block_size] < dist_xy;
                                    scalar_yz_closest_int = (1 - distance_check_2_mask) & (1 - distance_check_3_mask);

                                    xy_reduction_int += scalar_xz_closest_int + scalar_yz_closest_int;
                                    buffer_conflict_yz_block_int[z + y * block_size] += scalar_xy_closest_int + scalar_xz_closest_int;
                                    buffer_conflict_xz_block_int[z + x * block_size] += scalar_xy_closest_int + scalar_yz_closest_int;
                                }
                                buffer_conflict_xy_block_int[y + x * block_size] += xy_reduction_int;
                            }
                            // conflict_yz_block += n;
                        }
                    }

                }
                conflict_loop_time += omp_get_wtime() - time_start;
                time_start2 = omp_get_wtime();
                // conflict_xy_block = conflict_matrix + yb + xb * n;
                // conflict_xz_block = conflict_matrix + zb + xb * n;
                // conflict_yz_block = conflict_matrix + zb + yb * n;
                conflict_xz_block_int = conflict_matrix_int + zb + xb * n;
                conflict_yz_block_int = conflict_matrix_int + zb + yb * n;

                for(i = 0; i < x_block; ++i){
                    for(j = 0; j < z_block; ++j){
                        // conflict_xz_block[j + i * n] += buffer_conflict_xz_block[j + i * block_size];
                        // conflict_yz_block[j + i * n] += buffer_conflict_yz_block[j + i * block_size];
                        conflict_xz_block_int[j + i * n] += buffer_conflict_xz_block_int[j + i * block_size];
                        // conflict_yz_block_int[j + i * n] += buffer_conflict_yz_block_int[j + i * block_size];
                    }
                }
                for(i = 0; i < y_block; ++i){
                    for(j = 0; j < z_block; ++j){
                        // conflict_xz_block[j + i * n] += buffer_conflict_xz_block[j + i * block_size];
                        // conflict_yz_block[j + i * n] += buffer_conflict_yz_block[j + i * block_size];
                        // conflict_xz_block_int[j + i * n] += buffer_conflict_xz_block_int[j + i * block_size];
                        conflict_yz_block_int[j + i * n] += buffer_conflict_yz_block_int[j + i * block_size];
                    }

                }
                memops_loop_time += omp_get_wtime() - time_start2;
                // printf("(xb: %d, yb: %d, zb: %d)\n", xb, yb, zb);
                // print_matrix_int(block_size, block_size, buffer_conflict_yz_block_int);
                // printf("[\n");
                // conflict_xy_block_int = conflict_matrix_int + yb + xb * n;
                // for(i = 0; i < block_size; ++i){
                //     for(j = 0; j < block_size; ++j){
                //         printf("%d ", conflict_xy_block_int[j + i * n] + buffer_conflict_xy_block_int[j + i * block_size]);
                //     }
                //     printf(";\n");
                // }
                // printf("];\n");

            }
            time_start2 = omp_get_wtime();
            // conflict_xy_block = conflict_matrix + yb + xb * n;
            conflict_xy_block_int = conflict_matrix_int + yb + xb * n;
            for(i = 0; i < x_block; ++i){
                for(j = 0; j < y_block; ++j){
                    // conflict_xy_block[j + i * n] += buffer_conflict_xy_block[j + i * block_size];
                    conflict_xy_block_int[j + i * n] += buffer_conflict_xy_block_int[j + i * block_size];
                }
                // conflict_xy_block += n;
            }
            // print_matrix_int(n,n, conflict_matrix_int);
            // printf("(xb: %d, yb: %d)\n", xb, yb);
            // print_matrix_int(block_size, block_size, buffer_conflict_xy_block_int);
            memops_loop_time += omp_get_wtime() - time_start2;
        }
    }
    // print_matrix_int(n, n, conflict_matrix_int);
    // return;
    time_start = omp_get_wtime();
    for(i = 0; i < n * n; ++i){
        // conflict_matrix[i] = 1.f/conflict_matrix[i];
        conflict_matrix[i] = 1.f/conflict_matrix_int[i];
    }
    // print_matrix_int(n, n, conflict_matrix_int);
    conflict_loop_time += omp_get_wtime() - time_start;
    // return;
    // printf("\n\n");
        // initialize diagonal of C.
    float sum;
    time_start = omp_get_wtime();
    for (i = 0; i < n; ++i){
        sum = 0.f;
        for (j = 0; j < i; ++j){
            sum += conflict_matrix[i + j * n];
        }
        for (j = i + 1; j < n; ++j){
            sum += conflict_matrix[j + i * n];
        }
        C[i + i * n] = sum;
    }
    cohesion_loop_time += omp_get_wtime() - time_start;
    iters = 0;

    // print_matrix_int(n, n, conflict_matrix_int);
    time_start = omp_get_wtime();
    _mm_free(conflict_matrix_int);
    _mm_free(buffer_conflict_xy_block_int); _mm_free(buffer_conflict_xz_block_int); _mm_free(buffer_conflict_yz_block_int);
    _mm_free(distance_xy_block); _mm_free(distance_xz_block); _mm_free(distance_yz_block);
    float *conflict_xy_block, *conflict_xz_block, *conflict_yz_block;
    float *cohesion_xy_block ;
    float *cohesion_yx_block;
    float *cohesion_xz_block;
    float *cohesion_zx_block;
    float *cohesion_yz_block;
    float *cohesion_zy_block;
    block_size/=2;

    // float* restrict mask_xy_closest = (float *) _mm_malloc(block_size * sizeof(float), VECALIGN);
    // float* restrict mask_xz_closest = (float *) _mm_malloc(block_size * sizeof(float), VECALIGN);
    // float* restrict mask_yz_closest = (float *) _mm_malloc(block_size * sizeof(float), VECALIGN);

    float scalar_xy_closest, scalar_xz_closest, scalar_yz_closest;
    // float* restrict mask_tie_xy_xz = (float *) _mm_malloc(block_size * sizeof(float), VECALIGN);
    // float* restrict mask_tie_xy_yz = (float *) _mm_malloc(block_size * sizeof(float), VECALIGN);
    // float* restrict mask_tie_xz_yz = (float *) _mm_malloc(block_size * sizeof(float), VECALIGN);

    float* buffer_conflict_xz_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
    float* buffer_conflict_yz_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
    float* buffer_conflict_xy_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);

    float* restrict buffer_zx_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
    float* restrict buffer_zy_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
    float* restrict buffer_xz_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
    float* restrict buffer_yz_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
    float* restrict buffer_xy_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
    float* restrict buffer_yx_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);

    distance_xy_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
    distance_xz_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
    distance_yz_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);

    float* tmp_buffer_conflict_yz_block = buffer_conflict_yz_block;
    float* tmp_distance_yz_block = distance_yz_block;
    float* tmp_buffer_yz_block = buffer_yz_block;
    float* tmp_buffer_zy_block = buffer_zy_block;
    memops_loop_time += omp_get_wtime() - time_start;

    for(xb = 0; xb < n; xb += block_size){
        x_block = ((xb + block_size) < n) ? block_size : (n - xb);
        for(yb = xb; yb < n; yb += block_size){
            y_block = ((yb + block_size) < n) ? block_size : (n - yb);
            time_start = omp_get_wtime();
            for (i = 0; i < x_block; ++i){
                memcpy(distance_xy_block + i * block_size, D + yb + (xb + i) * n, sizeof(float)*y_block);
            }

            memset(buffer_yx_block,0,sizeof(float)*block_size*block_size);
            memset(buffer_xy_block,0,sizeof(float)*block_size*block_size);
            memops_loop_time += omp_get_wtime() - time_start;
            for(zb = yb; zb < n; zb += block_size){
                z_block = ((zb + block_size) < n) ? block_size : (n - zb);
                time_start = omp_get_wtime();
                conflict_xy_block = conflict_matrix + yb + xb * n;
                conflict_xz_block = conflict_matrix + zb + xb * n;
                conflict_yz_block = conflict_matrix + zb + yb * n;
                if(xb == yb){
                    #pragma unroll(8)
                    for (i = 0; i < x_block; ++i){
                        memcpy(distance_xz_block + i * block_size, D + zb + (xb + i) * n, sizeof(float)*z_block);
                        memcpy(buffer_conflict_xz_block + i * block_size, conflict_xz_block + i * n, sizeof(float)*z_block);
                        // memcpy(distance_yz_block + i * block_size, D + zb + (yb + i) * n, sizeof(float)*block_size);
                        // memcpy(buffer_conflict_yz_block + i * block_size, conflict_yz_block + i * n, sizeof(float)*block_size);
                    }
                    distance_yz_block = distance_xz_block;
                    buffer_conflict_yz_block = buffer_conflict_xz_block;
                    memset(buffer_zx_block,0,sizeof(float)*block_size*block_size);
                    memset(buffer_xz_block,0,sizeof(float)*block_size*block_size);

                    buffer_zy_block = buffer_zx_block;
                    buffer_yz_block = buffer_xz_block;
                }
                else{
                    distance_yz_block = tmp_distance_yz_block;
                    buffer_conflict_yz_block = tmp_buffer_conflict_yz_block;
                    buffer_zy_block = tmp_buffer_zy_block;
                    buffer_yz_block = tmp_buffer_yz_block;
                    #pragma unroll(8)
                    for (i = 0; i < x_block; ++i){
                        memcpy(distance_xz_block + i * block_size, D + zb + (xb + i) * n, sizeof(float)*z_block);
                        memcpy(buffer_conflict_xz_block + i * block_size, conflict_xz_block + i * n, sizeof(float)*z_block);
                        // memcpy(distance_yz_block + i * block_size, D + zb + (yb + i) * n, sizeof(float)*block_size);
                        // memcpy(buffer_conflict_yz_block + i * block_size, conflict_yz_block + i * n, sizeof(float)*block_size);
                    }
                    #pragma unroll(8)
                    for (i = 0; i < y_block; ++i){
                        // memcpy(distance_xz_block + i * block_size, D + zb + (xb + i) * n, sizeof(float)*block_size);
                        // memcpy(buffer_conflict_xz_block + i * block_size, conflict_xz_block + i * n, sizeof(float)*block_size);
                        memcpy(distance_yz_block + i * block_size, D + zb + (yb + i) * n, sizeof(float)*z_block);
                        memcpy(buffer_conflict_yz_block + i * block_size, conflict_yz_block + i * n, sizeof(float)*z_block);
                    }

                    memset(buffer_zx_block,0,sizeof(float)*block_size*block_size);
                    memset(buffer_zy_block,0,sizeof(float)*block_size*block_size);
                    memset(buffer_xz_block,0,sizeof(float)*block_size*block_size);
                    memset(buffer_yz_block,0,sizeof(float)*block_size*block_size);
                }
                memops_loop_time += omp_get_wtime() - time_start;

                time_start = omp_get_wtime();
                xend = x_block;
                // ystart = 0;
                // zstart = 0;
                if(xb == yb && yb == zb){
                    xend = x_block - 1;
                }
                for(x = 0; x < xend; ++x){
                    // if(xb == yb){
                    //     ystart = x + 1;

                    // }
                    ystart = (xb == yb) ? x + 1 : 0;
                    for(y = ystart; y < y_block; ++y){
                        xy_reduction = 0.f; yx_reduction = 0.f;
                        // if(yb == zb){
                        //     zstart = y + 1;
                        // }
                            // zstart = (yb == zb) ? y + 1 : 0;
                        dist_xy = distance_xy_block[y + x * block_size];
                        // loop_len = block_size - zstart;
                        if(yb == zb){
                            loop_len = z_block - y - 1;
                            conflict_xy_val = conflict_xy_block[y];
                            // for (z = y + 1; z < block_size; ++z){
                            for (z = 0; z < loop_len; ++z){
                                //compute masks for conflict blocks.
                                idx = z + y + 1;
                                distance_check_1_mask = dist_xy < distance_xz_block[idx + x * block_size];
                                distance_check_2_mask = dist_xy < distance_yz_block[idx + y * block_size];
                                scalar_xy_closest = distance_check_1_mask & distance_check_2_mask;

                                distance_check_1_mask = distance_xz_block[idx + x * block_size] < dist_xy;
                                distance_check_2_mask =  distance_xz_block[idx + x * block_size] < distance_yz_block[idx + y * block_size];
                                scalar_xz_closest = distance_check_1_mask & distance_check_2_mask;

                                distance_check_1_mask = distance_yz_block[idx + y * block_size] < distance_xz_block[idx + x * block_size];
                                distance_check_2_mask = distance_yz_block[idx + y * block_size] < dist_xy;
                                scalar_yz_closest = distance_check_1_mask & distance_check_2_mask;

                                // xy closest pair.
                                yx_reduction += scalar_xy_closest*buffer_conflict_xz_block[idx + x * block_size];
                                xy_reduction += scalar_xy_closest*buffer_conflict_yz_block[idx + y * block_size];

                                // xz closest pair.
                                buffer_xz_block[idx + x * block_size] += scalar_xz_closest*buffer_conflict_yz_block[idx + y * block_size];
                                buffer_zx_block[idx + x * block_size] += scalar_xz_closest*conflict_xy_val;

                                // yz closest pair.
                                buffer_yz_block[idx + y * block_size] += scalar_yz_closest*buffer_conflict_xz_block[idx + x * block_size];
                                buffer_zy_block[idx + y * block_size] += scalar_yz_closest*conflict_xy_val;
                            }
                            buffer_xy_block[y + x * block_size] += xy_reduction;
                            buffer_yx_block[y + x * block_size] += yx_reduction;
                        }
                        else{
                            conflict_xy_val = conflict_xy_block[y];
                            // __m512 all_ones = _mm512_set1_ps(1.f);
                            // __m512 dist_xy_avx = _mm512_set1_ps(dist_xy);
                            // __m512 conflict_xy_avx = _mm512_set1_ps(conflict_xy_val);

                            // __m512 xy_reduction_avx = _mm512_setzero();
                            // __m512 yx_reduction_avx = _mm512_setzero();
                            // __m512 dist_xz_avx, dist_yz_avx, conflict_xz_avx, conflict_yz_avx;
                            // __m512 cohesion_xz_avx, cohesion_zx_avx, cohesion_yz_avx, cohesion_zy_avx;
                            // __mmask16 mask_xy_closest, mask_xz_closest, mask_yz_closest;

                            // __m512 xy_reduction_avx_1 = _mm512_setzero();
                            // __m512 yx_reduction_avx_1 = _mm512_setzero();
                            // __m512 dist_xz_avx_1, dist_yz_avx_1, conflict_xz_avx_1, conflict_yz_avx_1;
                            // __m512 cohesion_xz_avx_1, cohesion_zx_avx_1, cohesion_yz_avx_1, cohesion_zy_avx_1;
                            // __mmask16 mask_xy_closest_1, mask_xz_closest_1, mask_yz_closest_1;

                            // __m512 xy_reduction_avx_2 = _mm512_setzero();
                            // __m512 yx_reduction_avx_2 = _mm512_setzero();
                            // __m512 dist_xz_avx_2, dist_yz_avx_2, conflict_xz_avx_2, conflict_yz_avx_2;
                            // __m512 cohesion_xz_avx_2, cohesion_zx_avx_2, cohesion_yz_avx_2, cohesion_zy_avx_2;
                            // __mmask16 mask_xy_closest_2, mask_xz_closest_2, mask_yz_closest_2;

                            // __m512 xy_reduction_avx_3 = _mm512_setzero();
                            // __m512 yx_reduction_avx_3 = _mm512_setzero();
                            // __m512 dist_xz_avx_3, dist_yz_avx_3, conflict_xz_avx_3, conflict_yz_avx_3;
                            // __m512 cohesion_xz_avx_3, cohesion_zx_avx_3, cohesion_yz_avx_3, cohesion_zy_avx_3;
                            // __mmask16 mask_xy_closest_3, mask_xz_closest_3, mask_yz_closest_3;

                            // __mmask16 distance_check_3_mask, distance_check_4_mask;
                            // __mmask16 distance_check_5_mask, distance_check_6_mask;
                            // __mmask16 distance_check_7_mask, distance_check_8_mask;

                            // for(z = 0; z < block_size; z += 128){
                            //     dist_xz_avx = _mm512_load_ps(distance_xz_block + z + x * block_size);
                            //     dist_yz_avx = _mm512_load_ps(distance_yz_block + z + y * block_size);
                            //     cohesion_xz_avx = _mm512_load_ps(buffer_xz_block + z + x * block_size);
                            //     cohesion_zx_avx = _mm512_load_ps(buffer_zx_block + z + x * block_size);
                            //     cohesion_yz_avx = _mm512_load_ps(buffer_yz_block + z + y * block_size);
                            //     cohesion_zy_avx = _mm512_load_ps(buffer_zy_block + z + y * block_size);
                            //     conflict_yz_avx = _mm512_load_ps(buffer_conflict_yz_block + z + y * block_size);
                            //     conflict_xz_avx = _mm512_load_ps(buffer_conflict_xz_block + z + x * block_size);

                            //     dist_xz_avx_1 = _mm512_load_ps(distance_xz_block + z + 16 + x * block_size);
                            //     dist_yz_avx_1 = _mm512_load_ps(distance_yz_block + z + 16 + y * block_size);
                            //     cohesion_xz_avx_1 = _mm512_load_ps(buffer_xz_block + z + 16 + x * block_size);
                            //     cohesion_zx_avx_1 = _mm512_load_ps(buffer_zx_block + z + 16 + x * block_size);
                            //     cohesion_yz_avx_1 = _mm512_load_ps(buffer_yz_block + z + 16 + y * block_size);
                            //     cohesion_zy_avx_1 = _mm512_load_ps(buffer_zy_block + z + 16 + y * block_size);
                            //     conflict_yz_avx_1 = _mm512_load_ps(buffer_conflict_yz_block + z + 16 + y * block_size);
                            //     conflict_xz_avx_1 = _mm512_load_ps(buffer_conflict_xz_block + z + 16 + x * block_size);

                            //     dist_xz_avx_2 = _mm512_load_ps(distance_xz_block + z + 32 + x * block_size);
                            //     dist_yz_avx_2 = _mm512_load_ps(distance_yz_block + z + 32 + y * block_size);
                            //     cohesion_xz_avx_2 = _mm512_load_ps(buffer_xz_block + z + 32 + x * block_size);
                            //     cohesion_zx_avx_2 = _mm512_load_ps(buffer_zx_block + z + 32 + x * block_size);
                            //     cohesion_yz_avx_2 = _mm512_load_ps(buffer_yz_block + z + 32 + y * block_size);
                            //     cohesion_zy_avx_2 = _mm512_load_ps(buffer_zy_block + z + 32 + y * block_size);
                            //     conflict_yz_avx_2 = _mm512_load_ps(buffer_conflict_yz_block + z + 32 + y * block_size);
                            //     conflict_xz_avx_2 = _mm512_load_ps(buffer_conflict_xz_block + z + 32 + x * block_size);

                            //     dist_xz_avx_3 = _mm512_load_ps(distance_xz_block + z + 64 + x * block_size);
                            //     dist_yz_avx_3 = _mm512_load_ps(distance_yz_block + z + 64 + y * block_size);
                            //     cohesion_xz_avx_3 = _mm512_load_ps(buffer_xz_block + z + 64 + x * block_size);
                            //     cohesion_zx_avx_3 = _mm512_load_ps(buffer_zx_block + z + 64 + x * block_size);
                            //     cohesion_yz_avx_3 = _mm512_load_ps(buffer_yz_block + z + 64 + y * block_size);
                            //     cohesion_zy_avx_3 = _mm512_load_ps(buffer_zy_block + z + 64 + y * block_size);
                            //     conflict_yz_avx_3 = _mm512_load_ps(buffer_conflict_yz_block + z + 64 + y * block_size);
                            //     conflict_xz_avx_3 = _mm512_load_ps(buffer_conflict_xz_block + z + 64 + x * block_size);

                            //     distance_check_1_mask = _mm512_cmplt_ps_mask(dist_xy_avx, dist_xz_avx);
                            //     distance_check_2_mask = _mm512_cmplt_ps_mask(dist_xy_avx, dist_yz_avx);
                            //     mask_xy_closest = distance_check_1_mask & distance_check_2_mask;

                            //     distance_check_1_mask = _mm512_cmplt_ps_mask(dist_xz_avx, dist_xy_avx);
                            //     distance_check_2_mask = _mm512_cmplt_ps_mask(dist_xz_avx, dist_yz_avx);
                            //     mask_xz_closest = distance_check_1_mask & distance_check_2_mask;

                            //     distance_check_1_mask = _mm512_cmplt_ps_mask(dist_yz_avx, dist_xz_avx);
                            //     distance_check_2_mask = _mm512_cmplt_ps_mask(dist_yz_avx, dist_xy_avx);
                            //     mask_yz_closest = distance_check_1_mask & distance_check_2_mask;

                            //     distance_check_3_mask = _mm512_cmplt_ps_mask(dist_xy_avx, dist_xz_avx);
                            //     distance_check_4_mask = _mm512_cmplt_ps_mask(dist_xy_avx, dist_yz_avx);
                            //     mask_xy_closest_1 = distance_check_3_mask & distance_check_4_mask;

                            //     distance_check_3_mask = _mm512_cmplt_ps_mask(dist_xz_avx, dist_xy_avx);
                            //     distance_check_4_mask = _mm512_cmplt_ps_mask(dist_xz_avx, dist_yz_avx);
                            //     mask_xz_closest_1 = distance_check_3_mask & distance_check_4_mask;

                            //     distance_check_3_mask = _mm512_cmplt_ps_mask(dist_yz_avx, dist_xz_avx);
                            //     distance_check_4_mask = _mm512_cmplt_ps_mask(dist_yz_avx, dist_xy_avx);
                            //     mask_yz_closest_1 = distance_check_3_mask & distance_check_4_mask;

                            //     distance_check_5_mask = _mm512_cmplt_ps_mask(dist_xy_avx, dist_xz_avx);
                            //     distance_check_6_mask = _mm512_cmplt_ps_mask(dist_xy_avx, dist_yz_avx);
                            //     mask_xy_closest_2 = distance_check_5_mask & distance_check_6_mask;

                            //     distance_check_5_mask = _mm512_cmplt_ps_mask(dist_xz_avx, dist_xy_avx);
                            //     distance_check_6_mask = _mm512_cmplt_ps_mask(dist_xz_avx, dist_yz_avx);
                            //     mask_xz_closest_2 = distance_check_5_mask & distance_check_6_mask;

                            //     distance_check_5_mask = _mm512_cmplt_ps_mask(dist_yz_avx, dist_xz_avx);
                            //     distance_check_6_mask = _mm512_cmplt_ps_mask(dist_yz_avx, dist_xy_avx);
                            //     mask_yz_closest_2 = distance_check_5_mask & distance_check_6_mask;

                            //     distance_check_7_mask = _mm512_cmplt_ps_mask(dist_xy_avx, dist_xz_avx);
                            //     distance_check_8_mask = _mm512_cmplt_ps_mask(dist_xy_avx, dist_yz_avx);
                            //     mask_xy_closest_3 = distance_check_7_mask & distance_check_8_mask;

                            //     distance_check_7_mask = _mm512_cmplt_ps_mask(dist_xz_avx, dist_xy_avx);
                            //     distance_check_8_mask = _mm512_cmplt_ps_mask(dist_xz_avx, dist_yz_avx);
                            //     mask_xz_closest_3 = distance_check_7_mask & distance_check_8_mask;

                            //     distance_check_7_mask = _mm512_cmplt_ps_mask(dist_yz_avx, dist_xz_avx);
                            //     distance_check_8_mask = _mm512_cmplt_ps_mask(dist_yz_avx, dist_xy_avx);
                            //     mask_yz_closest_3 = distance_check_7_mask & distance_check_8_mask;


                            //     xy_reduction_avx = _mm512_mask_fmadd_ps(all_ones, mask_xy_closest, conflict_yz_avx, xy_reduction_avx);
                            //     yx_reduction_avx = _mm512_mask_fmadd_ps(all_ones, mask_xy_closest, conflict_xz_avx, yx_reduction_avx);

                            //     xy_reduction_avx_1 = _mm512_mask_fmadd_ps(all_ones, mask_xy_closest_1, conflict_yz_avx_1, xy_reduction_avx_1);
                            //     yx_reduction_avx_1 = _mm512_mask_fmadd_ps(all_ones, mask_xy_closest_1, conflict_xz_avx_1, yx_reduction_avx_1);

                            //     xy_reduction_avx_2 = _mm512_mask_fmadd_ps(all_ones, mask_xy_closest_2, conflict_yz_avx_2, xy_reduction_avx_2);
                            //     yx_reduction_avx_2 = _mm512_mask_fmadd_ps(all_ones, mask_xy_closest_2, conflict_xz_avx_2, yx_reduction_avx_2);

                            //     xy_reduction_avx_3 = _mm512_mask_fmadd_ps(all_ones, mask_xy_closest_3, conflict_yz_avx_3, xy_reduction_avx_3);
                            //     yx_reduction_avx_3 = _mm512_mask_fmadd_ps(all_ones, mask_xy_closest_3, conflict_xz_avx_3, yx_reduction_avx_3);

                            //     _mm512_store_ps(buffer_xz_block + z + x * block_size, _mm512_mask_fmadd_ps(all_ones, mask_xz_closest, conflict_yz_avx, cohesion_xz_avx));
                            //     _mm512_store_ps(buffer_zx_block + z + x * block_size, _mm512_mask_fmadd_ps(all_ones, mask_xz_closest, conflict_xy_avx, cohesion_zx_avx));
                            //     _mm512_store_ps(buffer_yz_block + z + y * block_size, _mm512_mask_fmadd_ps(all_ones, mask_yz_closest, conflict_xz_avx, cohesion_yz_avx));
                            //     _mm512_store_ps(buffer_zy_block + z + y * block_size, _mm512_mask_fmadd_ps(all_ones, mask_yz_closest, conflict_xy_avx, cohesion_zy_avx));

                            //     _mm512_store_ps(buffer_xz_block + z + 16 + x * block_size, _mm512_mask_fmadd_ps(all_ones, mask_xz_closest_1, conflict_yz_avx_1, cohesion_xz_avx_1));
                            //     _mm512_store_ps(buffer_zx_block + z + 16 + x * block_size, _mm512_mask_fmadd_ps(all_ones, mask_xz_closest_1, conflict_xy_avx, cohesion_zx_avx_1));
                            //     _mm512_store_ps(buffer_yz_block + z + 16 + y * block_size, _mm512_mask_fmadd_ps(all_ones, mask_yz_closest_1, conflict_xz_avx_1, cohesion_yz_avx_1));
                            //     _mm512_store_ps(buffer_zy_block + z + 16 + y * block_size, _mm512_mask_fmadd_ps(all_ones, mask_yz_closest_1, conflict_xy_avx, cohesion_zy_avx_1));

                            //     _mm512_store_ps(buffer_xz_block + z + 32 + x * block_size, _mm512_mask_fmadd_ps(all_ones, mask_xz_closest_2, conflict_yz_avx_2, cohesion_xz_avx_2));
                            //     _mm512_store_ps(buffer_zx_block + z + 32 + x * block_size, _mm512_mask_fmadd_ps(all_ones, mask_xz_closest_2, conflict_xy_avx, cohesion_zx_avx_2));
                            //     _mm512_store_ps(buffer_yz_block + z + 32 + y * block_size, _mm512_mask_fmadd_ps(all_ones, mask_yz_closest_2, conflict_xz_avx_2, cohesion_yz_avx_2));
                            //     _mm512_store_ps(buffer_zy_block + z + 32 + y * block_size, _mm512_mask_fmadd_ps(all_ones, mask_yz_closest_2, conflict_xy_avx, cohesion_zy_avx_2));

                            //     _mm512_store_ps(buffer_xz_block + z + 64 + x * block_size, _mm512_mask_fmadd_ps(all_ones, mask_xz_closest_3, conflict_yz_avx_3, cohesion_xz_avx_3));
                            //     _mm512_store_ps(buffer_zx_block + z + 64 + x * block_size, _mm512_mask_fmadd_ps(all_ones, mask_xz_closest_3, conflict_xy_avx, cohesion_zx_avx_3));
                            //     _mm512_store_ps(buffer_yz_block + z + 64 + y * block_size, _mm512_mask_fmadd_ps(all_ones, mask_yz_closest_3, conflict_xz_avx_3, cohesion_yz_avx_3));
                            //     _mm512_store_ps(buffer_zy_block + z + 64 + y * block_size, _mm512_mask_fmadd_ps(all_ones, mask_yz_closest_3, conflict_xy_avx, cohesion_zy_avx_3));
                            // }
                            // xy_reduction_avx += xy_reduction_avx_1 + xy_reduction_avx_2 + xy_reduction_avx_3;
                            // yx_reduction_avx += yx_reduction_avx_1 + yx_reduction_avx_2 + yx_reduction_avx_3;
                            // xy_reduction += _mm512_reduce_add_ps(xy_reduction_avx);
                            // yx_reduction += _mm512_reduce_add_ps(yx_reduction_avx);
                            #pragma unroll(8)
                            //update cohesion blocks.
                            for (z = 0; z < z_block; ++z){
                                distance_check_1_mask = dist_xy < distance_xz_block[z + x * block_size];
                                distance_check_2_mask = dist_xy < distance_yz_block[z + y * block_size];
                                scalar_xy_closest = distance_check_1_mask & distance_check_2_mask;

                                distance_check_1_mask = distance_xz_block[z + x * block_size] < dist_xy;
                                distance_check_2_mask =  distance_xz_block[z + x * block_size] < distance_yz_block[z + y * block_size];
                                scalar_xz_closest = distance_check_1_mask & distance_check_2_mask;

                                distance_check_1_mask = distance_yz_block[z + y * block_size] < distance_xz_block[z + x * block_size];
                                distance_check_2_mask = distance_yz_block[z + y * block_size] < dist_xy;
                                scalar_yz_closest = distance_check_1_mask & distance_check_2_mask;

                                // xy closest pair.
                                xy_reduction += scalar_xy_closest*buffer_conflict_yz_block[z + y * block_size];
                                yx_reduction += scalar_xy_closest*buffer_conflict_xz_block[z + x * block_size];

                                // xz closest pair
                                buffer_xz_block[z + x * block_size] += scalar_xz_closest*buffer_conflict_yz_block[z + y * block_size];
                                buffer_zx_block[z + x * block_size] += scalar_xz_closest*conflict_xy_val;

                                //yz closest pair.
                                buffer_yz_block[z + y * block_size] += scalar_yz_closest*buffer_conflict_xz_block[z + x * block_size];
                                buffer_zy_block[z + y * block_size] += scalar_yz_closest*conflict_xy_val;
                            }
                            buffer_xy_block[y + x * block_size] += xy_reduction;
                            buffer_yx_block[y + x * block_size] += yx_reduction;
                        }
                    }
                    conflict_xy_block += n;
                }
                cohesion_loop_time += omp_get_wtime() - time_start;
                time_start2 = omp_get_wtime();
                if(xb == yb){
                    cohesion_zx_block = C + xb + zb * n;
                    cohesion_xz_block = C + zb + xb * n;
                    for(i = 0; i < z_block; ++i){
                        for(j = 0; j < x_block; ++j){
                            // printf("idx: %d\n", n*j + i);
                            cohesion_zx_block[j] += buffer_zx_block[i + j * block_size];
                            // cohesion_yx_block[j] += buffer_yx_block[i + j * block_size];
                        }
                        cohesion_zx_block += n;
                    }
                    for(i = 0; i < x_block; ++i){
                        for(j = 0; j < z_block; ++j){
                            cohesion_xz_block[j] += buffer_xz_block[j + i * block_size];
                            // cohesion_xy_block[j] += buffer_xy_block[j + i * block_size];
                        }
                        cohesion_xz_block += n;
                    }

                }
                else{
                    cohesion_zx_block = C + xb + zb * n;
                    cohesion_zy_block = C + yb + zb * n;
                    cohesion_xz_block = C + zb + xb * n;
                    cohesion_yz_block = C + zb + yb * n;
                    for(i = 0; i < z_block; ++i){
                        for(j = 0; j < x_block; ++j){
                            // printf("idx: %d\n", n*j + i);
                            cohesion_zx_block[j] += buffer_zx_block[i + j * block_size];
                            // cohesion_zy_block[j] += buffer_zy_block[i + j * block_size];
                            // cohesion_yx_block[j] += buffer_yx_block[i + j * block_size];
                        }
                        cohesion_zx_block += n;
                        // cohesion_zy_block += n;
                    }
                    for(i = 0; i < z_block; ++i){
                        for(j = 0; j < y_block; ++j){
                            // printf("idx: %d\n", n*j + i);
                            // cohesion_zx_block[j] += buffer_zx_block[i + j * block_size];
                            cohesion_zy_block[j] += buffer_zy_block[i + j * block_size];
                            // cohesion_yx_block[j] += buffer_yx_block[i + j * block_size];
                        }
                        // cohesion_zx_block += n;
                        cohesion_zy_block += n;
                    }
                    for(i = 0; i < x_block; ++i){
                        for(j = 0; j < z_block; ++j){
                            cohesion_xz_block[j] += buffer_xz_block[j + i * block_size];
                            // cohesion_yz_block[j] += buffer_yz_block[j + i * block_size];
                            // cohesion_xy_block[j] += buffer_xy_block[j + i * block_size];
                        }
                        cohesion_xz_block += n;
                        // cohesion_yz_block += n;
                    }
                    for(i = 0; i < y_block; ++i){
                        for(j = 0; j < z_block; ++j){
                            // cohesion_xz_block[j] += buffer_xz_block[j + i * block_size];
                            cohesion_yz_block[j] += buffer_yz_block[j + i * block_size];
                            // cohesion_xy_block[j] += buffer_xy_block[j + i * block_size];
                        }
                        // cohesion_xz_block += n;
                        cohesion_yz_block += n;
                    }
                }

                memops_loop_time += omp_get_wtime() - time_start2;
            }
            time_start2 = omp_get_wtime();
            cohesion_xy_block = C + yb + xb * n;
            cohesion_yx_block = C + xb + yb * n;
            for(i = 0; i < y_block; ++i){
                for(j = 0; j < x_block; ++j){
                    // printf("idx: %d\n", n*j + i);
                    cohesion_yx_block[j] += buffer_yx_block[i + j * block_size];
                }
                cohesion_yx_block += n;
            }

            for(i = 0; i < x_block; ++i){
                for(j = 0; j < y_block; ++j){
                    cohesion_xy_block[j] += buffer_xy_block[j + i * block_size];
                }
                cohesion_xy_block += n;
            }
            memops_loop_time += omp_get_wtime() - time_start2;
        }
    }
    // print_matrix(n, n, C);

    printf("==============================================\n");
    printf("Seq. Triplet Fewer Compares Loop Times\n");
    printf("==============================================\n");

    printf("memops loop time: %.5fs\n", memops_loop_time);
    printf("conflict focus size loop time: %.5fs\n", conflict_loop_time);
    printf("cohesion matrix update loop time: %.5fs\n\n", cohesion_loop_time);

    _mm_free(distance_xy_block); _mm_free(distance_xz_block); _mm_free(tmp_distance_yz_block);
    // _mm_free(mask_tie_xy_xz); _mm_free(mask_tie_xy_yz); _mm_free(mask_tie_xz_yz);
    _mm_free(buffer_zx_block); _mm_free(tmp_buffer_zy_block); _mm_free(buffer_yx_block);
    _mm_free(buffer_xz_block); _mm_free(tmp_buffer_yz_block); _mm_free(buffer_xy_block);
    // _mm_free(mask_xy_closest); _mm_free(mask_xz_closest); _mm_free(mask_yz_closest);
    _mm_free(buffer_conflict_xz_block); _mm_free(tmp_buffer_conflict_yz_block); _mm_free(buffer_conflict_xy_block);
    _mm_free(conflict_matrix);
    // _mm_free(conflict_matrix_int);
    // _mm_free(mask_xy_closest_int); _mm_free(mask_xz_closest_int); _mm_free(mask_yz_closest_int);
    // _mm_free(buffer_conflict_xz_block_int); _mm_free(buffer_conflict_yz_block_int); _mm_free(buffer_conflict_xy_block_int);

}

void pald_triplet_largezblock(float *D, float beta, unsigned int n, float *C, unsigned int block_size, unsigned int z_block_size){
    //TODO: Optimized sequential triplet code.
    float* restrict conflict_matrix = (float *)  _mm_malloc(n * n * sizeof(float), VECALIGN);
    unsigned int* restrict conflict_matrix_int = (unsigned int*)  _mm_malloc(n * n * sizeof(unsigned int), VECALIGN);

    float* restrict distance_xy_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
    float* restrict distance_xz_block = (float *) _mm_malloc(block_size * z_block_size * sizeof(float), VECALIGN);
    float* restrict distance_yz_block = (float *) _mm_malloc(block_size * z_block_size * sizeof(float), VECALIGN);


    unsigned int scalar_xy_closest_int, scalar_xz_closest_int, scalar_yz_closest_int;

    unsigned int* restrict buffer_conflict_xz_block_int = (unsigned int *) _mm_malloc(block_size * z_block_size * sizeof(unsigned int), VECALIGN);
    unsigned int* restrict buffer_conflict_yz_block_int = (unsigned int *) _mm_malloc(block_size * z_block_size * sizeof(unsigned int), VECALIGN);
    unsigned int* restrict buffer_conflict_xy_block_int = (unsigned int *) _mm_malloc(block_size * block_size * sizeof(unsigned int), VECALIGN);

    unsigned char distance_check_1_mask, distance_check_2_mask, symmetric_block_mask;
    unsigned int xy_reduction_int;
    float dist_xy  = 0.f;
    float conflict_xy_val = 0.f;
    unsigned int loop_len = 0;

    unsigned int *conflict_xy_block_int, *conflict_xz_block_int, *conflict_yz_block_int;
    // char print_out = 0;
    double time_start = 0.0, time_start2 = 0.0;
    double memops_loop_time = 0.0, conflict_loop_time = 0.0, cohesion_loop_time = 0.0;
    unsigned int xb, yb, zb, x, y, z;
    unsigned int i, j, k;
    unsigned int x_block, y_block, z_block;
    // int size_xy = block_size, size_xz = block_size, size_yz = block_size;
    unsigned int xend, ystart, zstart;
    float xy_reduction = 0.f, yx_reduction = 0.f, cohesion_sum = 0.f;
    // compute conflict focus sizes.
    unsigned int iters = 0;
    unsigned int idx;
    time_start = omp_get_wtime();
    for (i = 0; i < n; ++i){
        for (j = i + 1; j < n; ++j){
            // conflict_matrix[j + i * n] = 2.;
            conflict_matrix_int[j + i * n] = 2;
        }
    }
    conflict_loop_time += omp_get_wtime() - time_start;
    // if(print_out)
    //     print_matrix(n, n, conflict_matrix);


    //TODO: Add another level of blocking.
    for(xb = 0; xb < n; xb += block_size){
        x_block = ((xb + block_size) < n) ? block_size : (n - xb);
        for(yb = xb; yb < n; yb += block_size){
            y_block = ((yb + block_size) < n) ? block_size : (n - yb);
            time_start = omp_get_wtime();
            for (i = 0; i < x_block; ++i){
                memcpy(distance_xy_block + i * block_size, D + yb + (xb + i) * n, sizeof(float)*y_block);
            }

            // memset(buffer_conflict_xy_block, 0, sizeof(float)*block_size*block_size);
            memset(buffer_conflict_xy_block_int, 0, sizeof(int)*block_size*block_size);
            memops_loop_time += omp_get_wtime() - time_start;

            // copy DXY block from D.
            for(zb = yb; zb < n; zb += z_block_size){
                z_block = ((zb + z_block_size) < n) ? z_block_size : (n - zb);
                //copy DXZ and DYZ blocks from D.
                time_start = omp_get_wtime();

                for (i = 0; i < x_block; ++i){
                    memcpy(distance_xz_block + i * z_block_size, D + zb + (xb + i) * n, sizeof(float)*z_block);
                }
                for(i = 0; i < y_block; ++i){
                    memcpy(distance_yz_block + i * z_block_size, D + zb + (yb + i) * n, sizeof(float)*z_block);
                }

                // memset(buffer_conflict_xz_block, 0, sizeof(float)*block_size*block_size);
                // memset(buffer_conflict_yz_block, 0, sizeof(float)*block_size*block_size);
                memset(buffer_conflict_xz_block_int, 0, sizeof(int)*z_block_size*block_size);
                memset(buffer_conflict_yz_block_int, 0, sizeof(int)*z_block_size*block_size);
                memops_loop_time += omp_get_wtime() - time_start;

                time_start = omp_get_wtime();

                xend = (xb == yb && yb == zb) ? x_block - 1 : x_block;
                // ystart = 0;
                // zstart = 0;
                // if(xb == yb && yb == zb){
                //     xend = block_size - 1;
                // }

                for(x = 0; x < xend; ++x){
                    // if(xb == yb){
                    //     ystart = x + 1;
                    //     // conflict_yz_block += ystart*n;
                    // }
                    ystart = (xb == yb) ? x + 1 : 0;
                    // if(xb == yb){
                        // _mm_prefetch((float*) (distance_xz_block + x*z_block_size + 16*64), _MM_HINT_NTA);
                        // _mm_prefetch((unsigned int*) (buffer_conflict_xz_block_int + x*z_block_size + 64), _MM_HINT_NTA);
                        for(y = ystart; y < y_block; ++y){
                            // if(yb == zb){
                            //     zstart = y + 1;
                            // }
                            // xy_reduction = 0.f;
                            // contains_tie = 0.f;
                            // _mm_prefetch((unsigned int*) (buffer_conflict_yz_block_int + y*z_block_size + 64), _MM_HINT_T1);
                            if(yb == zb){
                                xy_reduction_int = 0;
                                // zstart = (yb == zb) ? y + 1 : 0;
                                dist_xy = distance_xy_block[y + x * block_size];
                                // for (z = y + 1; z < block_size; ++z){
                                idx = y + 1;
                                loop_len = z_block - idx;
                                #pragma unroll(16)
                                for (z = 0; z < loop_len; ++z){
                                    //compute masks for conflict blocks.
                                    distance_check_1_mask = dist_xy < distance_xz_block[idx + x * z_block_size];
                                    distance_check_2_mask = dist_xy < distance_yz_block[idx + y * z_block_size];
                                    scalar_xy_closest_int = distance_check_1_mask & distance_check_2_mask;

                                    distance_check_1_mask = distance_xz_block[idx + x * z_block_size] < dist_xy;
                                    distance_check_2_mask = distance_xz_block[idx + x *z_block_size] < distance_yz_block[idx + y * z_block_size];
                                    scalar_xz_closest_int = distance_check_1_mask & distance_check_2_mask;

                                    distance_check_1_mask = distance_yz_block[idx + y * z_block_size] < distance_xz_block[idx + x * z_block_size];
                                    distance_check_2_mask = distance_yz_block[idx + y * z_block_size] < dist_xy;
                                    scalar_yz_closest_int = distance_check_1_mask & distance_check_2_mask;

                                    xy_reduction_int += scalar_xz_closest_int + scalar_yz_closest_int;
                                    buffer_conflict_yz_block_int[idx + y * z_block_size] += scalar_xy_closest_int + scalar_xz_closest_int;
                                    buffer_conflict_xz_block_int[idx + x * z_block_size] += scalar_xy_closest_int + scalar_yz_closest_int;
                                    idx++;
                                }
                                buffer_conflict_xy_block_int[y + x * block_size] += xy_reduction_int;

                                // #pragma unroll(16)
                                // for (z = 0; z < z_block; ++z){
                                //     //compute masks for conflict blocks.
                                //     symmetric_block_mask = ~(z < (y + 1));
                                //     distance_check_1_mask = dist_xy < distance_xz_block[z + x * z_block_size];
                                //     distance_check_2_mask = dist_xy < distance_yz_block[z + y * z_block_size];
                                //     scalar_xy_closest_int = symmetric_block_mask & distance_check_1_mask & distance_check_2_mask;

                                //     distance_check_1_mask = distance_xz_block[z + x * z_block_size] < dist_xy;
                                //     distance_check_2_mask = distance_xz_block[z + x *z_block_size] < distance_yz_block[z + y * z_block_size];
                                //     scalar_xz_closest_int = symmetric_block_mask & distance_check_1_mask & distance_check_2_mask;

                                //     distance_check_1_mask = distance_yz_block[z + y * z_block_size] < distance_xz_block[z + x * z_block_size];
                                //     distance_check_2_mask = distance_yz_block[z + y * z_block_size] < dist_xy;
                                //     scalar_yz_closest_int = symmetric_block_mask & distance_check_1_mask & distance_check_2_mask;


                                //     xy_reduction_int += scalar_xz_closest_int + scalar_yz_closest_int;
                                //     buffer_conflict_yz_block_int[z + y * z_block_size] += scalar_xy_closest_int + scalar_xz_closest_int;
                                //     buffer_conflict_xz_block_int[z + x * z_block_size] += scalar_xy_closest_int + scalar_yz_closest_int;
                                // }
                                // buffer_conflict_xy_block_int[y + x * block_size] += xy_reduction_int;
                            }
                            else{
                                xy_reduction_int = 0;
                                // zstart = (yb == zb) ? y + 1 : 0;
                                dist_xy = distance_xy_block[y + x * block_size];
                                __assume_aligned(distance_xz_block, VECALIGN);
                                __assume_aligned(distance_yz_block, VECALIGN);
                                __assume_aligned(buffer_conflict_xz_block_int, VECALIGN);
                                __assume_aligned(buffer_conflict_yz_block_int, VECALIGN);

                                #pragma unroll(16)
                                for (z = 0; z < z_block; ++z){
                                    //compute masks for conflict blocks.
                                    distance_check_1_mask = dist_xy < distance_xz_block[z + x * z_block_size];
                                    distance_check_2_mask = dist_xy < distance_yz_block[z + y * z_block_size];
                                    scalar_xy_closest_int = distance_check_1_mask & distance_check_2_mask;

                                    distance_check_1_mask = distance_xz_block[z + x * z_block_size] < dist_xy;
                                    distance_check_2_mask = distance_xz_block[z + x * z_block_size] < distance_yz_block[z + y * z_block_size];
                                    scalar_xz_closest_int = distance_check_1_mask & distance_check_2_mask;

                                    distance_check_1_mask = distance_yz_block[z + y * z_block_size] < distance_xz_block[z + x * z_block_size];
                                    distance_check_2_mask = distance_yz_block[z + y * z_block_size] < dist_xy;
                                    scalar_yz_closest_int = distance_check_1_mask & distance_check_2_mask;

                                    xy_reduction_int += scalar_xz_closest_int + scalar_yz_closest_int;
                                    buffer_conflict_yz_block_int[z + y * z_block_size] += scalar_xy_closest_int + scalar_xz_closest_int;
                                    buffer_conflict_xz_block_int[z + x * z_block_size] += scalar_xy_closest_int + scalar_yz_closest_int;
                                }
                                buffer_conflict_xy_block_int[y + x * block_size] += xy_reduction_int;
                                // _mm_prefetch((unsigned int*) (buffer_conflict_xz_block_int + (x+1)*z_block_size), _MM_HINT_NTA);
                                // _mm_prefetch((unsigned int*) (buffer_conflict_yz_block_int + (y+1)*z_block_size), _MM_HINT_NTA);
                                // _mm_prefetch((float*) (distance_xz_block + (x+1)*z_block_size), _MM_HINT_NTA);
                                // _mm_prefetch((float*) (distance_yz_block + (y+1)*z_block_size), _MM_HINT_NTA);
                            }
                            // conflict_yz_block += n;
                        }
                    // }
                    // else{
                    //     for(y = 0; y < y_block; ++y){
                    //         // if(yb == zb){
                    //         //     zstart = y + 1;
                    //         // }
                    //         // xy_reduction = 0.f;
                    //         // contains_tie = 0.f;
                    //         if(yb == zb){
                    //             xy_reduction_int = 0;
                    //             // zstart = (yb == zb) ? y + 1 : 0;
                    //             dist_xy = distance_xy_block[y + x * block_size];

                    //             idx = y + 1;
                    //             loop_len = z_block - idx;
                    //             #pragma unroll(16)
                    //             for (z = 0; z < loop_len; ++z){
                    //                 //compute masks for conflict blocks.

                    //                 distance_check_1_mask = dist_xy < distance_xz_block[idx + x * z_block_size];
                    //                 distance_check_2_mask = dist_xy < distance_yz_block[idx + y * z_block_size];
                    //                 scalar_xy_closest_int = distance_check_1_mask & distance_check_2_mask;

                    //                 distance_check_1_mask = distance_xz_block[idx + x * z_block_size] < dist_xy;
                    //                 distance_check_2_mask =  distance_xz_block[idx + x * z_block_size] < distance_yz_block[idx + y * z_block_size];
                    //                 scalar_xz_closest_int = distance_check_1_mask & distance_check_2_mask;

                    //                 distance_check_1_mask = distance_yz_block[idx + y * z_block_size] < distance_xz_block[idx + x * z_block_size];
                    //                 distance_check_2_mask = distance_yz_block[idx + y * z_block_size] < dist_xy;
                    //                 scalar_yz_closest_int = distance_check_1_mask & distance_check_2_mask;


                    //                 xy_reduction_int += scalar_xz_closest_int + scalar_yz_closest_int;
                    //                 buffer_conflict_yz_block_int[idx + y * z_block_size] += scalar_xy_closest_int + scalar_xz_closest_int;
                    //                 buffer_conflict_xz_block_int[idx + x * z_block_size] += scalar_xy_closest_int + scalar_yz_closest_int;

                    //                 idx++;
                    //             }
                    //             buffer_conflict_xy_block_int[y + x * block_size] += xy_reduction_int;
                    //         }
                    //         else{
                    //             xy_reduction_int = 0;
                    //             // zstart = (yb == zb) ? y + 1 : 0;
                    //             dist_xy = distance_xy_block[y + x * block_size];

                    //             #pragma unroll(16)
                    //             for (z = 0; z < z_block; ++z){
                    //                 //compute masks for conflict blocks.
                    //                 distance_check_1_mask = dist_xy < distance_xz_block[z + x * z_block_size];
                    //                 distance_check_2_mask = dist_xy < distance_yz_block[z + y * z_block_size];
                    //                 scalar_xy_closest_int = distance_check_1_mask & distance_check_2_mask;

                    //                 distance_check_1_mask = distance_xz_block[z + x * z_block_size] < dist_xy;
                    //                 distance_check_2_mask = distance_xz_block[z + x * z_block_size] < distance_yz_block[z + y * z_block_size];
                    //                 scalar_xz_closest_int = distance_check_1_mask & distance_check_2_mask;

                    //                 distance_check_1_mask = distance_yz_block[z + y * z_block_size] < distance_xz_block[z + x * z_block_size];
                    //                 distance_check_2_mask = distance_yz_block[z + y * z_block_size] < dist_xy;
                    //                 scalar_yz_closest_int = distance_check_1_mask & distance_check_2_mask;

                    //                 xy_reduction_int += scalar_xz_closest_int + scalar_yz_closest_int;
                    //                 buffer_conflict_yz_block_int[z + y * z_block_size] += scalar_xy_closest_int + scalar_xz_closest_int;
                    //                 buffer_conflict_xz_block_int[z + x * z_block_size] += scalar_xy_closest_int + scalar_yz_closest_int;
                    //             }
                    //             buffer_conflict_xy_block_int[y + x * block_size] += xy_reduction_int;
                    //         }
                    //         // conflict_yz_block += n;
                    //     }
                    // }

                }
                conflict_loop_time += omp_get_wtime() - time_start;
                time_start2 = omp_get_wtime();
                // conflict_xy_block = conflict_matrix + yb + xb * n;
                // conflict_xz_block = conflict_matrix + zb + xb * n;
                // conflict_yz_block = conflict_matrix + zb + yb * n;
                conflict_xz_block_int = conflict_matrix_int + zb + xb * n;
                conflict_yz_block_int = conflict_matrix_int + zb + yb * n;

                for(i = 0; i < x_block; ++i){
                    for(j = 0; j < z_block; ++j){
                        // conflict_xz_block[j + i * n] += buffer_conflict_xz_block[j + i * block_size];
                        // conflict_yz_block[j + i * n] += buffer_conflict_yz_block[j + i * block_size];
                        conflict_xz_block_int[j + i * n] += buffer_conflict_xz_block_int[j + i * z_block_size];
                        // conflict_yz_block_int[j + i * n] += buffer_conflict_yz_block_int[j + i * block_size];
                    }
                }
                for(i = 0; i < y_block; ++i){
                    for(j = 0; j < z_block; ++j){
                        // conflict_xz_block[j + i * n] += buffer_conflict_xz_block[j + i * block_size];
                        // conflict_yz_block[j + i * n] += buffer_conflict_yz_block[j + i * block_size];
                        // conflict_xz_block_int[j + i * n] += buffer_conflict_xz_block_int[j + i * block_size];
                        conflict_yz_block_int[j + i * n] += buffer_conflict_yz_block_int[j + i * z_block_size];
                    }

                }
                memops_loop_time += omp_get_wtime() - time_start2;
                // printf("(xb: %d, yb: %d, zb: %d)\n", xb, yb, zb);
                // print_matrix_int(block_size, block_size, buffer_conflict_yz_block_int);
                // printf("[\n");
                // conflict_xy_block_int = conflict_matrix_int + yb + xb * n;
                // for(i = 0; i < block_size; ++i){
                //     for(j = 0; j < block_size; ++j){
                //         printf("%d ", conflict_xy_block_int[j + i * n] + buffer_conflict_xy_block_int[j + i * block_size]);
                //     }
                //     printf(";\n");
                // }
                // printf("];\n");

            }
            time_start2 = omp_get_wtime();
            // conflict_xy_block = conflict_matrix + yb + xb * n;
            conflict_xy_block_int = conflict_matrix_int + yb + xb * n;
            for(i = 0; i < x_block; ++i){
                for(j = 0; j < y_block; ++j){
                    // conflict_xy_block[j + i * n] += buffer_conflict_xy_block[j + i * block_size];
                    conflict_xy_block_int[j + i * n] += buffer_conflict_xy_block_int[j + i * block_size];
                }
                // conflict_xy_block += n;
            }
            // print_matrix_int(n,n, conflict_matrix_int);
            // printf("(xb: %d, yb: %d)\n", xb, yb);
            // print_matrix_int(block_size, block_size, buffer_conflict_xy_block_int);
            memops_loop_time += omp_get_wtime() - time_start2;
        }
    }
    // print_matrix_int(n, n, conflict_matrix_int);
    // return;
    time_start = omp_get_wtime();
    for(i = 0; i < n * n; ++i){
        // conflict_matrix[i] = 1.f/conflict_matrix[i];
        conflict_matrix[i] = 1.f/conflict_matrix_int[i];
    }
    // print_matrix_int(n, n, conflict_matrix_int);
    conflict_loop_time += omp_get_wtime() - time_start;
    // return;
    // printf("\n\n");
        // initialize diagonal of C.
    float sum;
    time_start = omp_get_wtime();
    for (i = 0; i < n; ++i){
        sum = 0.f;
        for (j = 0; j < i; ++j){
            sum += conflict_matrix[i + j * n];
        }
        for (j = i + 1; j < n; ++j){
            sum += conflict_matrix[j + i * n];
        }
        C[i + i * n] = sum;
    }
    cohesion_loop_time += omp_get_wtime() - time_start;
    iters = 0;

    // print_matrix_int(n, n, conflict_matrix_int);
    time_start = omp_get_wtime();
    _mm_free(conflict_matrix_int);
    _mm_free(buffer_conflict_xy_block_int); _mm_free(buffer_conflict_xz_block_int); _mm_free(buffer_conflict_yz_block_int);
    _mm_free(distance_xy_block); _mm_free(distance_xz_block); _mm_free(distance_yz_block);
    float *conflict_xy_block, *conflict_xz_block, *conflict_yz_block;
    float *cohesion_xy_block ;
    float *cohesion_yx_block;
    float *cohesion_xz_block;
    float *cohesion_zx_block;
    float *cohesion_yz_block;
    float *cohesion_zy_block;
    block_size/=2;

    // float* restrict mask_xy_closest = (float *) _mm_malloc(block_size * sizeof(float), VECALIGN);
    // float* restrict mask_xz_closest = (float *) _mm_malloc(block_size * sizeof(float), VECALIGN);
    // float* restrict mask_yz_closest = (float *) _mm_malloc(block_size * sizeof(float), VECALIGN);

    float scalar_xy_closest, scalar_xz_closest, scalar_yz_closest;

    // float* restrict mask_tie_xy_xz = (float *) _mm_malloc(block_size * sizeof(float), VECALIGN);
    // float* restrict mask_tie_xy_yz = (float *) _mm_malloc(block_size * sizeof(float), VECALIGN);
    // float* restrict mask_tie_xz_yz = (float *) _mm_malloc(block_size * sizeof(float), VECALIGN);

    float* buffer_conflict_xz_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
    float* buffer_conflict_yz_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
    float* buffer_conflict_xy_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);

    float* restrict buffer_zx_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
    float* restrict buffer_zy_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
    float* restrict buffer_xz_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
    float* restrict buffer_yz_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
    float* restrict buffer_xy_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
    float* restrict buffer_yx_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);

    distance_xy_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
    distance_xz_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
    distance_yz_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);

    float* tmp_buffer_conflict_yz_block = buffer_conflict_yz_block;
    float* tmp_distance_yz_block = distance_yz_block;
    float* tmp_buffer_yz_block = buffer_yz_block;
    float* tmp_buffer_zy_block = buffer_zy_block;
    memops_loop_time += omp_get_wtime() - time_start;

    for(xb = 0; xb < n; xb += block_size){
        x_block = ((xb + block_size) < n) ? block_size : (n - xb);
        for(yb = xb; yb < n; yb += block_size){
            y_block = ((yb + block_size) < n) ? block_size : (n - yb);
            time_start = omp_get_wtime();
            for (i = 0; i < x_block; ++i){
                memcpy(distance_xy_block + i * block_size, D + yb + (xb + i) * n, sizeof(float)*y_block);
            }

            memset(buffer_yx_block,0,sizeof(float)*block_size*block_size);
            memset(buffer_xy_block,0,sizeof(float)*block_size*block_size);
            memops_loop_time += omp_get_wtime() - time_start;
            for(zb = yb; zb < n; zb += block_size){
                z_block = ((zb + block_size) < n) ? block_size : (n - zb);
                time_start = omp_get_wtime();
                conflict_xy_block = conflict_matrix + yb + xb * n;
                conflict_xz_block = conflict_matrix + zb + xb * n;
                conflict_yz_block = conflict_matrix + zb + yb * n;
                if(xb == yb){
                    #pragma unroll(8)
                    for (i = 0; i < x_block; ++i){
                        memcpy(distance_xz_block + i * block_size, D + zb + (xb + i) * n, sizeof(float)*z_block);
                        memcpy(buffer_conflict_xz_block + i * block_size, conflict_xz_block + i * n, sizeof(float)*z_block);
                        // memcpy(distance_yz_block + i * block_size, D + zb + (yb + i) * n, sizeof(float)*block_size);
                        // memcpy(buffer_conflict_yz_block + i * block_size, conflict_yz_block + i * n, sizeof(float)*block_size);
                    }
                    distance_yz_block = distance_xz_block;
                    buffer_conflict_yz_block = buffer_conflict_xz_block;
                    memset(buffer_zx_block,0,sizeof(float)*block_size*block_size);
                    memset(buffer_xz_block,0,sizeof(float)*block_size*block_size);

                    buffer_zy_block = buffer_zx_block;
                    buffer_yz_block = buffer_xz_block;
                }
                else{
                    distance_yz_block = tmp_distance_yz_block;
                    buffer_conflict_yz_block = tmp_buffer_conflict_yz_block;
                    buffer_zy_block = tmp_buffer_zy_block;
                    buffer_yz_block = tmp_buffer_yz_block;
                    #pragma unroll(8)
                    for (i = 0; i < x_block; ++i){
                        memcpy(distance_xz_block + i * block_size, D + zb + (xb + i) * n, sizeof(float)*z_block);
                        memcpy(buffer_conflict_xz_block + i * block_size, conflict_xz_block + i * n, sizeof(float)*z_block);
                        // memcpy(distance_yz_block + i * block_size, D + zb + (yb + i) * n, sizeof(float)*block_size);
                        // memcpy(buffer_conflict_yz_block + i * block_size, conflict_yz_block + i * n, sizeof(float)*block_size);
                    }
                    #pragma unroll(8)
                    for (i = 0; i < y_block; ++i){
                        // memcpy(distance_xz_block + i * block_size, D + zb + (xb + i) * n, sizeof(float)*block_size);
                        // memcpy(buffer_conflict_xz_block + i * block_size, conflict_xz_block + i * n, sizeof(float)*block_size);
                        memcpy(distance_yz_block + i * block_size, D + zb + (yb + i) * n, sizeof(float)*z_block);
                        memcpy(buffer_conflict_yz_block + i * block_size, conflict_yz_block + i * n, sizeof(float)*z_block);
                    }

                    memset(buffer_zx_block,0,sizeof(float)*block_size*block_size);
                    memset(buffer_zy_block,0,sizeof(float)*block_size*block_size);
                    memset(buffer_xz_block,0,sizeof(float)*block_size*block_size);
                    memset(buffer_yz_block,0,sizeof(float)*block_size*block_size);
                }
                memops_loop_time += omp_get_wtime() - time_start;

                time_start = omp_get_wtime();
                xend = x_block;
                // ystart = 0;
                // zstart = 0;
                if(xb == yb && yb == zb){
                    xend = x_block - 1;
                }
                for(x = 0; x < xend; ++x){
                    // if(xb == yb){
                    //     ystart = x + 1;

                    // }
                    ystart = (xb == yb) ? x + 1 : 0;
                    for(y = ystart; y < y_block; ++y){
                        xy_reduction = 0.f; yx_reduction = 0.f;
                        // if(yb == zb){
                        //     zstart = y + 1;
                        // }
                            // zstart = (yb == zb) ? y + 1 : 0;
                        dist_xy = distance_xy_block[y + x * block_size];
                        // loop_len = block_size - zstart;
                        if(yb == zb){
                            idx = y + 1;
                            loop_len = z_block - y - 1;
                            conflict_xy_val = conflict_xy_block[y];
                            // for (z = y + 1; z < block_size; ++z){
                            #pragma nounroll
                            for (z = 0; z < loop_len; ++z){
                                //compute masks for conflict blocks.
                                distance_check_1_mask = dist_xy < distance_xz_block[idx + x * block_size];
                                distance_check_2_mask = dist_xy < distance_yz_block[idx + y * block_size];
                                scalar_xy_closest = distance_check_1_mask & distance_check_2_mask;

                                distance_check_1_mask = distance_xz_block[idx + x * block_size] < dist_xy;
                                distance_check_2_mask =  distance_xz_block[idx + x * block_size] < distance_yz_block[idx + y * block_size];
                                scalar_xz_closest = distance_check_1_mask & distance_check_2_mask;

                                distance_check_1_mask = distance_yz_block[idx + y * block_size] < distance_xz_block[idx + x * block_size];
                                distance_check_2_mask = distance_yz_block[idx + y * block_size] < dist_xy;
                                scalar_yz_closest = distance_check_1_mask & distance_check_2_mask;

                                // xy closest pair.
                                yx_reduction += scalar_xy_closest*buffer_conflict_xz_block[idx + x * block_size];
                                xy_reduction += scalar_xy_closest*buffer_conflict_yz_block[idx + y * block_size];

                                // xz closest pair.
                                buffer_xz_block[idx + x * block_size] += scalar_xz_closest*buffer_conflict_yz_block[idx + y * block_size];
                                buffer_zx_block[idx + x * block_size] += scalar_xz_closest*conflict_xy_val;

                                // yz closest pair.
                                buffer_yz_block[idx + y * block_size] += scalar_yz_closest*buffer_conflict_xz_block[idx + x * block_size];
                                buffer_zy_block[idx + y * block_size] += scalar_yz_closest*conflict_xy_val;
                                idx++;
                            }
                            buffer_xy_block[y + x * block_size] += xy_reduction;
                            buffer_yx_block[y + x * block_size] += yx_reduction;
                        }
                        else{
                            conflict_xy_val = conflict_xy_block[y];
                            __assume_aligned(distance_xz_block, VECALIGN);
                            __assume_aligned(distance_yz_block, VECALIGN);

                            __assume_aligned(buffer_xz_block, VECALIGN);
                            __assume_aligned(buffer_yz_block, VECALIGN);

                            __assume_aligned(buffer_zx_block, VECALIGN);
                            __assume_aligned(buffer_zy_block, VECALIGN);

                            #pragma unroll(8)
                            //update cohesion blocks.
                            for (z = 0; z < z_block; ++z){
                                distance_check_1_mask = dist_xy < distance_xz_block[z + x * block_size];
                                distance_check_2_mask = dist_xy < distance_yz_block[z + y * block_size];
                                scalar_xy_closest = distance_check_1_mask & distance_check_2_mask;

                                distance_check_1_mask = distance_xz_block[z + x * block_size] < dist_xy;
                                distance_check_2_mask =  distance_xz_block[z + x * block_size] < distance_yz_block[z + y * block_size];
                                scalar_xz_closest = distance_check_1_mask & distance_check_2_mask;

                                distance_check_1_mask = distance_yz_block[z + y * block_size] < distance_xz_block[z + x * block_size];
                                distance_check_2_mask = distance_yz_block[z + y * block_size] < dist_xy;
                                scalar_yz_closest = distance_check_1_mask & distance_check_2_mask;

                                // xy closest pair.
                                xy_reduction += scalar_xy_closest*buffer_conflict_yz_block[z + y * block_size];
                                yx_reduction += scalar_xy_closest*buffer_conflict_xz_block[z + x * block_size];

                                // xz closest pair
                                buffer_xz_block[z + x * block_size] += scalar_xz_closest*buffer_conflict_yz_block[z + y * block_size];
                                buffer_zx_block[z + x * block_size] += scalar_xz_closest*conflict_xy_val;

                                //yz closest pair.
                                buffer_yz_block[z + y * block_size] += scalar_yz_closest*buffer_conflict_xz_block[z + x * block_size];
                                buffer_zy_block[z + y * block_size] += scalar_yz_closest*conflict_xy_val;
                            }
                            buffer_xy_block[y + x * block_size] += xy_reduction;
                            buffer_yx_block[y + x * block_size] += yx_reduction;
                        }
                    }
                    conflict_xy_block += n;
                }
                cohesion_loop_time += omp_get_wtime() - time_start;
                time_start2 = omp_get_wtime();
                if(xb == yb){
                    cohesion_zx_block = C + xb + zb * n;
                    cohesion_xz_block = C + zb + xb * n;
                    for(i = 0; i < z_block; ++i){
                        for(j = 0; j < x_block; ++j){
                            // printf("idx: %d\n", n*j + i);
                            cohesion_zx_block[j] += buffer_zx_block[i + j * block_size];
                            // cohesion_yx_block[j] += buffer_yx_block[i + j * block_size];
                        }
                        cohesion_zx_block += n;
                    }
                    for(i = 0; i < x_block; ++i){
                        for(j = 0; j < z_block; ++j){
                            cohesion_xz_block[j] += buffer_xz_block[j + i * block_size];
                            // cohesion_xy_block[j] += buffer_xy_block[j + i * block_size];
                        }
                        cohesion_xz_block += n;
                    }

                }
                else{
                    cohesion_zx_block = C + xb + zb * n;
                    cohesion_zy_block = C + yb + zb * n;
                    cohesion_xz_block = C + zb + xb * n;
                    cohesion_yz_block = C + zb + yb * n;
                    for(i = 0; i < z_block; ++i){
                        for(j = 0; j < x_block; ++j){
                            // printf("idx: %d\n", n*j + i);
                            cohesion_zx_block[j] += buffer_zx_block[i + j * block_size];
                            // cohesion_zy_block[j] += buffer_zy_block[i + j * block_size];
                            // cohesion_yx_block[j] += buffer_yx_block[i + j * block_size];
                        }
                        cohesion_zx_block += n;
                        // cohesion_zy_block += n;
                    }
                    for(i = 0; i < z_block; ++i){
                        for(j = 0; j < y_block; ++j){
                            // printf("idx: %d\n", n*j + i);
                            // cohesion_zx_block[j] += buffer_zx_block[i + j * block_size];
                            cohesion_zy_block[j] += buffer_zy_block[i + j * block_size];
                            // cohesion_yx_block[j] += buffer_yx_block[i + j * block_size];
                        }
                        // cohesion_zx_block += n;
                        cohesion_zy_block += n;
                    }
                    for(i = 0; i < x_block; ++i){
                        for(j = 0; j < z_block; ++j){
                            cohesion_xz_block[j] += buffer_xz_block[j + i * block_size];
                            // cohesion_yz_block[j] += buffer_yz_block[j + i * block_size];
                            // cohesion_xy_block[j] += buffer_xy_block[j + i * block_size];
                        }
                        cohesion_xz_block += n;
                        // cohesion_yz_block += n;
                    }
                    for(i = 0; i < y_block; ++i){
                        for(j = 0; j < z_block; ++j){
                            // cohesion_xz_block[j] += buffer_xz_block[j + i * block_size];
                            cohesion_yz_block[j] += buffer_yz_block[j + i * block_size];
                            // cohesion_xy_block[j] += buffer_xy_block[j + i * block_size];
                        }
                        // cohesion_xz_block += n;
                        cohesion_yz_block += n;
                    }
                }

                memops_loop_time += omp_get_wtime() - time_start2;
            }
            time_start2 = omp_get_wtime();
            cohesion_xy_block = C + yb + xb * n;
            cohesion_yx_block = C + xb + yb * n;
            for(i = 0; i < y_block; ++i){
                for(j = 0; j < x_block; ++j){
                    // printf("idx: %d\n", n*j + i);
                    cohesion_yx_block[j] += buffer_yx_block[i + j * block_size];
                }
                cohesion_yx_block += n;
            }

            for(i = 0; i < x_block; ++i){
                for(j = 0; j < y_block; ++j){
                    cohesion_xy_block[j] += buffer_xy_block[j + i * block_size];
                }
                cohesion_xy_block += n;
            }
            memops_loop_time += omp_get_wtime() - time_start2;
        }
    }
    // print_matrix(n, n, C);

    printf("==============================================\n");
    printf("Seq. Triplet large z-block Loop Times\n");
    printf("==============================================\n");

    printf("memops loop time: %.5fs\n", memops_loop_time);
    printf("conflict focus size loop time: %.5fs\n", conflict_loop_time);
    printf("cohesion matrix update loop time: %.5fs\n\n", cohesion_loop_time);

    _mm_free(distance_xy_block); _mm_free(distance_xz_block); _mm_free(tmp_distance_yz_block);
    // _mm_free(mask_tie_xy_xz); _mm_free(mask_tie_xy_yz); _mm_free(mask_tie_xz_yz);
    _mm_free(buffer_zx_block); _mm_free(tmp_buffer_zy_block); _mm_free(buffer_yx_block);
    _mm_free(buffer_xz_block); _mm_free(tmp_buffer_yz_block); _mm_free(buffer_xy_block);
    // _mm_free(mask_xy_closest); _mm_free(mask_xz_closest); _mm_free(mask_yz_closest);
    _mm_free(buffer_conflict_xz_block); _mm_free(tmp_buffer_conflict_yz_block); _mm_free(buffer_conflict_xy_block);
    _mm_free(conflict_matrix);
    // _mm_free(conflict_matrix_int);
    // _mm_free(mask_xy_closest_int); _mm_free(mask_xz_closest_int); _mm_free(mask_yz_closest_int);
    // _mm_free(buffer_conflict_xz_block_int); _mm_free(buffer_conflict_yz_block_int); _mm_free(buffer_conflict_xy_block_int);

}

void pald_triplet(float* restrict D, float beta, int n, float* restrict C, int block_size){
    //TODO: Optimized sequential triplet code.
    float* conflict_matrix = (float *)  _mm_malloc(n * n * sizeof(float), VECALIGN);
    memset(conflict_matrix, 0, n * n * sizeof(float));

    float* restrict distance_xy_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
    float* restrict distance_xz_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
    float* restrict distance_yz_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);

    float* restrict mask_xy_closest = (float *) _mm_malloc(block_size * sizeof(float), VECALIGN);
    float* restrict mask_xz_closest = (float *) _mm_malloc(block_size * sizeof(float), VECALIGN);
    float* restrict mask_yz_closest = (float *) _mm_malloc(block_size * sizeof(float), VECALIGN);

    float* restrict mask_tie_xy_xz = (float *) _mm_malloc(block_size * sizeof(float), VECALIGN);
    float* restrict mask_tie_xy_yz = (float *) _mm_malloc(block_size * sizeof(float), VECALIGN);
    float* restrict mask_tie_xz_yz = (float *) _mm_malloc(block_size * sizeof(float), VECALIGN);

    float* restrict buffer_zx_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
    float* restrict buffer_zy_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
    float* restrict buffer_xz_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
    float* restrict buffer_yz_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
    float* restrict buffer_xy_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
    float* restrict buffer_yx_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);

    float* restrict buffer_conflict_xz_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
    float* restrict buffer_conflict_yz_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
    float* restrict buffer_conflict_xy_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
    // float* restrict buffer_contains_tie = (float *) _mm_malloc(block_size * sizeof(float), VECALIGN);

    char distance_check_1 = 0;
    char distance_check_2 = 0;
    char distance_check_3 = 0;
    float contains_tie = 0.f;
    float alpha = 0.f;
    float dist_xy  = 0.f;
    float conflict_xy_val = 0.f;
    float loop_len = 0.f;

    float *conflict_xy_block, *conflict_xz_block, *conflict_yz_block;
    float *cohesion_xy_block ;
    float *cohesion_yx_block;
    float *cohesion_xz_block;
    float *cohesion_zx_block;
    float *cohesion_yz_block;
    float *cohesion_zy_block;
    // char print_out = 0;
    double time_start = 0.0, time_start2 = 0.0;
    double memops_loop_time = 0.0, conflict_loop_time = 0.0, cohesion_loop_time = 0.0;
    time_start = omp_get_wtime();
    for (int i = 0; i < n; ++i){
        for (int j = i + 1; j < n; ++j){
            conflict_matrix[j + i * n] = 2.;
        }
    }
    conflict_loop_time += omp_get_wtime() - time_start;
    // if(print_out)
    //     print_matrix(n, n, conflict_matrix);

    int xb, yb, zb, x, y, z;
    int i, j, k;
    // int size_xy = block_size, size_xz = block_size, size_yz = block_size;
    int xend, ystart, zstart;
    float xy_reduction = 0.f, yx_reduction = 0.f, cohesion_sum = 0.f;
    // compute conflict focus sizes.
    int iters = 0;
    //TODO: Add another level of blocking.
    for(xb = 0; xb < n; xb += block_size){
        for(yb = xb; yb < n; yb += block_size){
            time_start = omp_get_wtime();
            for (i = 0; i < block_size; ++i){
                memcpy(distance_xy_block + i * block_size, D + yb + (xb + i) * n, sizeof(float)*block_size);
            }

            memset(buffer_conflict_xy_block, 0, sizeof(float)*block_size*block_size);
            memops_loop_time += omp_get_wtime() - time_start;

            // copy DXY block from D.
            for(zb = yb; zb < n; zb += block_size){
                //copy DXZ and DYZ blocks from D.c
                time_start = omp_get_wtime();
                for (i = 0; i < block_size; ++i){

                    memcpy(distance_xz_block + i * block_size, D + zb + (xb + i) * n, sizeof(float)*block_size);
                }
                for(i = 0; i < block_size; ++i){
                    memcpy(distance_yz_block + i * block_size, D + zb + (yb + i) * n, sizeof(float)*block_size);
                }

                memset(buffer_conflict_xz_block, 0, sizeof(float)*block_size*block_size);
                memset(buffer_conflict_yz_block, 0, sizeof(float)*block_size*block_size);
                memops_loop_time += omp_get_wtime() - time_start;

                time_start = omp_get_wtime();

                xend = block_size;
                // ystart = 0;
                // zstart = 0;
                if(xb == yb && yb == zb){
                    xend = block_size - 1;
                }
                for(x = 0; x < xend; ++x){
                    // if(xb == yb){
                    //     ystart = x + 1;
                    //     // conflict_yz_block += ystart*n;
                    // }
                    ystart = (xb == yb) ? x + 1 : 0;
                    for(y = ystart; y < block_size; ++y){
                        // if(yb == zb){
                        //     zstart = y + 1;
                        // }
                        xy_reduction = 0.f;
                        zstart = (yb == zb) ? y + 1 : 0;
                        dist_xy = distance_xy_block[y + x * block_size];
                        contains_tie = 0.f;
                        loop_len = block_size - zstart;
                        for (z = zstart; z < block_size; ++z){
                            //compute masks for conflict blocks.
                            distance_check_1 = dist_xy < distance_xz_block[z + x * block_size];
                            distance_check_2 = dist_xy < distance_yz_block[z + y * block_size];
                            mask_xy_closest[z] = distance_check_1 & distance_check_2;

                            distance_check_1 = distance_xz_block[z + x * block_size] < dist_xy;
                            distance_check_2 =  distance_xz_block[z + x * block_size] < distance_yz_block[z + y * block_size];
                            mask_xz_closest[z] = distance_check_1 & distance_check_2;

                            distance_check_1 = distance_yz_block[z + y * block_size] < distance_xz_block[z + x * block_size];
                            distance_check_2 = distance_yz_block[z + y * block_size] < dist_xy;
                            mask_yz_closest[z] = distance_check_1 & distance_check_2;
                        }
                        for(z = zstart; z < block_size; ++z){
                            //contains_tie = (1.f + (-mask_xy_closest[z]))*(1.f + (-mask_xz_closest[z]))*(1.f + (-mask_yz_closest[z]));
                            xy_reduction += mask_xz_closest[z] + mask_yz_closest[z];
                            buffer_conflict_yz_block[z + y * block_size] += mask_xy_closest[z] + mask_xz_closest[z];
                            buffer_conflict_xz_block[z + x * block_size] += mask_xy_closest[z] + mask_yz_closest[z];
                            contains_tie += mask_xy_closest[z] + mask_xz_closest[z] + mask_yz_closest[z];
                        }
                        if(contains_tie < loop_len){
                            for(z = zstart; z < block_size; ++z){
                                contains_tie = (1.f + (-mask_xy_closest[z]))*(1.f + (-mask_xz_closest[z]))*(1.f + (-mask_yz_closest[z]));
                                xy_reduction += contains_tie;
                                buffer_conflict_yz_block[z + y * block_size] += contains_tie;
                                buffer_conflict_xz_block[z + x * block_size] += contains_tie;
                            }
                        }
                        // print_matrix(block_size, n, conflict_xy_block);
                        buffer_conflict_xy_block[y + x * block_size] += xy_reduction;
                        // conflict_yz_block += n;
                    }

                }
                conflict_loop_time += omp_get_wtime() - time_start;
                time_start2 = omp_get_wtime();
                conflict_xy_block = conflict_matrix + yb + xb * n;
                conflict_xz_block = conflict_matrix + zb + xb * n;
                conflict_yz_block = conflict_matrix + zb + yb * n;

                for(i = 0; i < block_size; ++i){
                    for(j = 0; j < block_size; ++j){
                        conflict_xz_block[j + i * n] += buffer_conflict_xz_block[j + i * block_size];
                        conflict_yz_block[j + i * n] += buffer_conflict_yz_block[j + i * block_size];
                    }

                }
                memops_loop_time += omp_get_wtime() - time_start2;

            }
            time_start2 = omp_get_wtime();
            conflict_xy_block = conflict_matrix + yb + xb * n;
            for(i = 0; i < block_size; ++i){
                for(j = 0; j < block_size; ++j){
                    conflict_xy_block[j + i * n] += buffer_conflict_xy_block[j + i * block_size];
                }
                // conflict_xy_block += n;
            }
            memops_loop_time += omp_get_wtime() - time_start2;
        }
    }
    time_start = omp_get_wtime();
    for(i = 0; i < n * n; ++i){
        conflict_matrix[i] = 1.f/conflict_matrix[i];
    }
    conflict_loop_time += omp_get_wtime() - time_start;
    // print_matrix(n, n, conflict_matrix);
    // return;
    // printf("\n\n");
        // initialize diagonal of C.
    float sum;
    time_start = omp_get_wtime();
    for (i = 0; i < n; ++i){
        sum = 0.f;
        for (j = 0; j < i; ++j){
            sum += conflict_matrix[i + j * n];
        }

        for (j = i + 1; j < n; ++j){
            sum += conflict_matrix[j + i * n];
        }
        C[i + i * n] = sum;
    }
    cohesion_loop_time += omp_get_wtime() - time_start;
    iters = 0;
    block_size/=2;
    for(xb = 0; xb < n; xb += block_size){
        for(yb = xb; yb < n; yb += block_size){
            time_start = omp_get_wtime();
            for (i = 0; i < block_size; ++i){
                memcpy(distance_xy_block + i * block_size, D + yb + (xb + i) * n, sizeof(float)*block_size);
            }

            memset(buffer_yx_block,0,sizeof(float)*block_size*block_size);
            memset(buffer_xy_block,0,sizeof(float)*block_size*block_size);
            memops_loop_time += omp_get_wtime() - time_start;
            for(zb = yb; zb < n; zb += block_size){
                time_start = omp_get_wtime();
                conflict_xy_block = conflict_matrix + yb + xb * n;
                conflict_xz_block = conflict_matrix + zb + xb * n;
                conflict_yz_block = conflict_matrix + zb + yb * n;
                for (i = 0; i < block_size; ++i){
                    memcpy(distance_xz_block + i * block_size, D + zb + (xb + i) * n, sizeof(float)*block_size);
                    memcpy(distance_yz_block + i * block_size, D + zb + (yb + i) * n, sizeof(float)*block_size);
                    memcpy(buffer_conflict_xz_block + i * block_size, conflict_xz_block + i * n, sizeof(float)*block_size);
                    memcpy(buffer_conflict_yz_block + i * block_size, conflict_yz_block + i * n, sizeof(float)*block_size);

                }

                memset(buffer_zx_block,0,sizeof(float)*block_size*block_size);
                memset(buffer_zy_block,0,sizeof(float)*block_size*block_size);
                memset(buffer_xz_block,0,sizeof(float)*block_size*block_size);
                memset(buffer_yz_block,0,sizeof(float)*block_size*block_size);

                memops_loop_time += omp_get_wtime() - time_start;

                time_start = omp_get_wtime();
                xend = block_size;
                ystart = 0;
                zstart = 0;

                if(xb == yb && yb == zb){
                    xend = block_size - 1;
                }
                for(x = 0; x < xend; ++x){
                    if(xb == yb){
                        ystart = x + 1;

                    }
                    for(y = ystart; y < block_size; ++y){
                        xy_reduction = 0.f; yx_reduction = 0.f; contains_tie = 0.f;
                        if(yb == zb){
                            zstart = y + 1;
                        }
                        dist_xy = distance_xy_block[y + x * block_size];
                        loop_len = block_size - zstart;
                        contains_tie = 0.f;
                        for (z = zstart; z < block_size; ++z){
                            //compute masks for conflict blocks.
                            distance_check_1 = dist_xy < distance_xz_block[z + x * block_size];
                            distance_check_2 = dist_xy < distance_yz_block[z + y * block_size];
                            mask_xy_closest[z] = distance_check_1 & distance_check_2;

                            distance_check_1 = distance_xz_block[z + x * block_size] < dist_xy;
                            distance_check_2 =  distance_xz_block[z + x * block_size] < distance_yz_block[z + y * block_size];
                            mask_xz_closest[z] = distance_check_1 & distance_check_2;

                            distance_check_1 = distance_yz_block[z + y * block_size] < distance_xz_block[z + x * block_size];
                            distance_check_2 = distance_yz_block[z + y * block_size] < dist_xy;
                            mask_yz_closest[z] = distance_check_1 & distance_check_2;
                            // contains_tie += (1.f + (-mask_xy_closest[z]))*(1.f + (-mask_xz_closest[z]))*(1.f + (-mask_yz_closest[z]));
                        }
                        conflict_xy_val = conflict_xy_block[y];
                        for(z = zstart; z < block_size; ++z){
                            //xy closest pair.
                            yx_reduction += mask_xy_closest[z]*buffer_conflict_xz_block[z + x * block_size];
                            xy_reduction += mask_xy_closest[z]*buffer_conflict_yz_block[z + y * block_size];

                            //xz closest pair.
                            // cohesion_xz_block[z] += mask_xz_closest[z]*conflict_yz_block[z];
                            buffer_xz_block[z + x * block_size] += mask_xz_closest[z]*buffer_conflict_yz_block[z + y * block_size];
                            buffer_zx_block[z + x * block_size] += mask_xz_closest[z]*conflict_xy_val;

                            // yz closest pair.
                            buffer_yz_block[z + y * block_size] += mask_yz_closest[z]*buffer_conflict_xz_block[z + x * block_size];
                            buffer_zy_block[z + y * block_size] += mask_yz_closest[z]*conflict_xy_val;

                            contains_tie += mask_xy_closest[z] + mask_xz_closest[z] + mask_yz_closest[z];
                        }

                        if(contains_tie < loop_len){
                            for(z = zstart; z < block_size; ++z){
                                mask_tie_xy_xz[z] = (distance_xy_block[y + x * block_size] == distance_xz_block[z + x * block_size]) ? 1.f : 0.f;
                                mask_tie_xy_yz[z] = (distance_xy_block[y + x * block_size] == distance_yz_block[z + y * block_size]) ? 1.f : 0.f;
                                mask_tie_xz_yz[z] = (distance_xz_block[z + x * block_size] == distance_yz_block[z + y * block_size]) ? 1.0f : 0.f;
                            }
                            for(z = zstart; z < block_size; ++z){
                                //xy closest pair.
                                alpha = (1.f + (-mask_xy_closest[z]))*(1.f + (-mask_xz_closest[z]))*(1.f + (-mask_yz_closest[z]));
                                yx_reduction += (alpha*mask_tie_xy_xz[z]*buffer_conflict_xz_block[z + x * block_size]);
                                yx_reduction += (0.5f*alpha*mask_tie_xy_yz[z])*buffer_conflict_xz_block[z + x * block_size];

                                xy_reduction += alpha*0.5f*mask_tie_xy_xz[z]*buffer_conflict_yz_block[z + y * block_size];
                                xy_reduction += alpha*mask_tie_xy_yz[z]*buffer_conflict_yz_block[z + y * block_size];

                                //xz closest pair.
                                cohesion_sum = alpha*0.5f*mask_tie_xy_xz[z]*buffer_conflict_yz_block[z + y * block_size];
                                cohesion_sum += alpha*mask_tie_xz_yz[z]*buffer_conflict_yz_block[z  + y * block_size];
                                buffer_xz_block[z + x * block_size] += cohesion_sum;

                                cohesion_sum = alpha*mask_tie_xy_xz[z]*conflict_xy_val;
                                cohesion_sum += alpha*0.5f*mask_tie_xz_yz[z]*conflict_xy_val;
                                buffer_zx_block[z + x * block_size] += cohesion_sum;

                                // yz closest pair.
                                cohesion_sum = alpha*.5f*mask_tie_xz_yz[z]*conflict_xy_val;
                                cohesion_sum += alpha*mask_tie_xy_yz[z]*conflict_xy_val;
                                buffer_zy_block[z + y * block_size] += cohesion_sum;

                                cohesion_sum = alpha*mask_tie_xz_yz[z]*buffer_conflict_xz_block[z + x * block_size];
                                cohesion_sum += alpha*0.5f*mask_tie_xy_yz[z]*buffer_conflict_xz_block[z + x * block_size];
                                buffer_yz_block[z + y * block_size] += cohesion_sum;

                            }
                        }
                        buffer_xy_block[y + x * block_size] += xy_reduction;
                        buffer_yx_block[y + x * block_size] += yx_reduction;

                    }
                    conflict_xy_block += n;
                }

                cohesion_zx_block = C + xb + zb * n;
                cohesion_zy_block = C + yb + zb * n;
                cohesion_xz_block = C + zb + xb * n;
                cohesion_yz_block = C + zb + yb * n;

                time_start2 = omp_get_wtime();

                for(i = 0; i < block_size; ++i){
                    for(j = 0; j < block_size; ++j){
                        // printf("idx: %d\n", n*j + i);
                        cohesion_zx_block[j] += buffer_zx_block[i + j * block_size];
                        cohesion_zy_block[j] += buffer_zy_block[i + j * block_size];
                        // cohesion_yx_block[j] += buffer_yx_block[i + j * block_size];
                    }
                    cohesion_zx_block += n;
                    cohesion_zy_block += n;
                }

                for(i = 0; i < block_size; ++i){
                    for(j = 0; j < block_size; ++j){
                        cohesion_xz_block[j] += buffer_xz_block[j + i * block_size];
                        cohesion_yz_block[j] += buffer_yz_block[j + i * block_size];
                        // cohesion_xy_block[j] += buffer_xy_block[j + i * block_size];
                    }
                    cohesion_xz_block += n;
                    cohesion_yz_block += n;
                }

                memops_loop_time += omp_get_wtime() - time_start2;
                cohesion_loop_time += omp_get_wtime() - time_start;
            }
            time_start2 = omp_get_wtime();
            cohesion_xy_block = C + yb + xb * n;
            cohesion_yx_block = C + xb + yb * n;
            for(i = 0; i < block_size; ++i){
                for(j = 0; j < block_size; ++j){
                    // printf("idx: %d\n", n*j + i);
                    cohesion_yx_block[j] += buffer_yx_block[i + j * block_size];
                }
                cohesion_yx_block += n;
            }
            cohesion_yx_block = C + xb + yb * n;

            conflict_xy_block = conflict_matrix + yb + xb * n;

            for(i = 0; i < block_size; ++i){
                for(j = 0; j < block_size; ++j){
                    cohesion_xy_block[j] += buffer_xy_block[j + i * block_size];
                }
                cohesion_xy_block += n;
            }
            memops_loop_time += omp_get_wtime() - time_start2;
        }
    }
    // print_matrix(n, n, C);

    printf("======================================\n");
    printf("Seq. Triplet Optimized Loop Times\n");
    printf("======================================\n");

    printf("memops loop time: %.5fs\n", memops_loop_time);
    printf("conflict focus size loop time: %.5fs\n", conflict_loop_time);
    printf("cohesion matrix update loop time: %.5fs\n\n", cohesion_loop_time);

    _mm_free(distance_xy_block); _mm_free(distance_xz_block); _mm_free(distance_yz_block);
    _mm_free(mask_tie_xy_xz); _mm_free(mask_tie_xy_yz); _mm_free(mask_tie_xz_yz);
    _mm_free(mask_xy_closest); _mm_free(mask_xz_closest); _mm_free(mask_yz_closest);
    _mm_free(buffer_zx_block); _mm_free(buffer_zy_block); _mm_free(buffer_yx_block);
    _mm_free(buffer_xz_block); _mm_free(buffer_yz_block); _mm_free(buffer_xy_block);
    _mm_free(buffer_conflict_xz_block); _mm_free(buffer_conflict_yz_block); _mm_free(buffer_conflict_xy_block);
    _mm_free(conflict_matrix);
}

void pald_triplet_L2_blocked(float* restrict D, const float beta, const int n, float* restrict C, int block_size, int l2_block_size){
   //TODO: Optimized sequential triplet code.
    unsigned int* restrict conflict_matrix_int = (unsigned int *)  _mm_malloc(n * n * sizeof(int), VECALIGN);
    memset(conflict_matrix_int, 0, n * n * sizeof(int));
    float* restrict distance_xy_block = (float *) _mm_malloc(block_size * block_size * sizeof(int), VECALIGN);
    float* restrict distance_xz_block = (float *) _mm_malloc(block_size * block_size * sizeof(int), VECALIGN);
    float* restrict distance_yz_block = (float *) _mm_malloc(block_size * block_size * sizeof(int), VECALIGN);

    // float* restrict distance_xy_L2_block = (float *) _mm_malloc(l2_block_size * l2_block_size * sizeof(float), VECALIGN);
    // float* restrict distance_xz_L2_block = (float *) _mm_malloc(l2_block_size * l2_block_size * sizeof(float), VECALIGN);
    // float* restrict distance_yz_L2_block = (float *) _mm_malloc(l2_block_size * l2_block_size * sizeof(float), VECALIGN);

    unsigned int* restrict buffer_conflict_xz_block_int = (unsigned int *) _mm_malloc(block_size * block_size * sizeof(int), VECALIGN);
    unsigned int* restrict buffer_conflict_yz_block_int = (unsigned int *) _mm_malloc(block_size * block_size * sizeof(int), VECALIGN);
    unsigned int* restrict buffer_conflict_xy_block_int = (unsigned int *) _mm_malloc(block_size * block_size * sizeof(int), VECALIGN);

    // unsigned int * restrict buffer_conflict_xz_L2_block_int = (unsigned int *) _mm_malloc(l2_block_size * l2_block_size * sizeof(int), VECALIGN);
    // unsigned int * restrict buffer_conflict_yz_L2_block_int = (unsigned int *) _mm_malloc(l2_block_size * l2_block_size * sizeof(int), VECALIGN);
    // unsigned int * restrict buffer_conflict_xy_L2_block_int = (unsigned int *) _mm_malloc(l2_block_size * l2_block_size * sizeof(int), VECALIGN);

    unsigned int distance_check_1 = 0, distance_check_2 = 0, distance_check_3 = 0;
    float dist_xy  = 0.f;
    float conflict_xy_val = 0.f;
    unsigned int loop_len;

    unsigned int xy_reduction_int = 0;
    unsigned int scalar_xy_closest_int, scalar_xz_closest_int, scalar_yz_closest_int;
    unsigned int *conflict_xy_block_int, *conflict_xz_block_int, *conflict_yz_block_int;

    double time_start = 0.0, time_start2 = 0.0;
    double memops_loop_time = 0.0, conflict_loop_time = 0.0, cohesion_loop_time = 0.0;
    time_start = omp_get_wtime();
    #pragma unroll_and_jam(16)
    for (int i = 0; i < n; ++i){
        for (int j = i + 1; j < n; ++j){
            conflict_matrix_int[j + i * n] = 2;
        }
    }
    conflict_loop_time += omp_get_wtime() - time_start;
    // if(print_out)
    //     print_matrix(n, n, conflict_matrix);


    unsigned int xb, yb, zb, x, y, z;
    unsigned int xbl2, ybl2, zbl2;
    unsigned int i, j, k, idx;
    // int size_xy = block_size, size_xz = block_size, size_yz = block_size;
    unsigned int xend, ystart, zstart;
    unsigned int ybstart, zbstart;
    // unsigned int * restrict conflict_xy_ptr;
    // unsigned int * restrict conflict_xz_ptr;
    // unsigned int * restrict conflict_yz_ptr;
    // compute conflict focus sizes.
    for(xbl2 = 0; xbl2 < n; xbl2 += l2_block_size){
        for(ybl2 = xbl2; ybl2 < n; ybl2 += l2_block_size){
            // for(i = 0; i < l2_block_size; ++i){
            //     memcpy(distance_xy_L2_block + i * l2_block_size, D + ybl2 + (xbl2 + i) * n, sizeof(float)*l2_block_size);
            // }
            // memset(buffer_conflict_xy_L2_block_int, 0, sizeof(int)*l2_block_size*l2_block_size);
            for(zbl2 = ybl2; zbl2 < n; zbl2 += l2_block_size){
                // for (i = 0; i < l2_block_size; ++i){
                //     memcpy(distance_xz_L2_block + i * l2_block_size, D + zbl2 + (xbl2 + i) * n, sizeof(float)*l2_block_size);
                // }
                // for (i = 0; i < l2_block_size; ++i){
                //     memcpy(distance_yz_L2_block + i * l2_block_size, D + zbl2 + (ybl2 + i) * n, sizeof(float)*l2_block_size);
                // }
                // memset(buffer_conflict_xz_L2_block_int, 0, sizeof(int)*l2_block_size*l2_block_size);
                // memset(buffer_conflict_yz_L2_block_int, 0, sizeof(int)*l2_block_size*l2_block_size);
                for(xb = 0; xb < l2_block_size; xb += block_size){
                    ybstart = (xbl2 == ybl2) ? (xb) : 0;
                    for(yb = ybstart; yb < l2_block_size; yb += block_size){
                        time_start = omp_get_wtime();
                        // // distance_xy_block = distance_xy_L2_block + yb + xb*l2_block_size;
                        for (i = 0; i < block_size; ++i){
                            //size_xy = (xb == yb) ? i : block_size;
                            // time_start = omp_get_wtime();
                            memcpy(distance_xy_block + i * block_size, D + ybl2 + yb + (xbl2 + xb + i) * n, sizeof(float)*block_size);
                        }
                        // distance_xy_block = distance_xy_L2_block + yb + xb * l2_block_size;
                        // conflict_xy_ptr = buffer_conflict_xy_L2_block_int + yb + xb * l2_block_size;
                        memset(buffer_conflict_xy_block_int, 0, sizeof(int)*block_size*block_size);
                        memops_loop_time += omp_get_wtime() - time_start;

                        // copy DXY block from D.
                        zbstart = (zbl2 == ybl2) ? (yb) : 0;
                        for(zb = zbstart; zb < l2_block_size; zb += block_size){
                            //copy DXZ and DYZ blocks from D.c
                            time_start = omp_get_wtime();
                            // distance_xz_block = distance_xz_L2_block + zb + xb * l2_block_size;
                            for (i = 0; i < block_size; ++i){
                                memcpy(distance_xz_block + i * block_size, D + zbl2 + zb + (xbl2 + xb + i) * n, sizeof(float)*block_size);
                            }
                            // distance_yz_block = distance_yz_L2_block + zb + yb * l2_block_size;
                            for (i = 0; i < block_size; ++i){
                                memcpy(distance_yz_block + i * block_size, D + zbl2 + zb + (ybl2 + yb + i) * n, sizeof(float)*block_size);
                            }
                            // conflict_xz_ptr = buffer_conflict_xz_L2_block_int + zb + xb * l2_block_size;
                            // conflict_yz_ptr = buffer_conflict_yz_L2_block_int + zb + yb * l2_block_size;

                            memset(buffer_conflict_xz_block_int, 0, sizeof(int)*block_size*block_size);
                            memset(buffer_conflict_yz_block_int, 0, sizeof(int)*block_size*block_size);
                            memops_loop_time += omp_get_wtime() - time_start;

                            time_start = omp_get_wtime();

                            xend = block_size;
                            // ystart = 0;
                            // zstart = 0;
                            if(xbl2 == ybl2 && ybl2 == zbl2 && xb == yb && yb == zb){
                                xend = block_size - 1;
                            }
                            for(x = 0; x < xend; ++x){
                                ystart = (xbl2 == ybl2 && xb == yb) ? x + 1 : 0;
                                for(y = ystart; y < block_size; ++y){
                                    // zstart = (ybl2 == zbl2 && yb == zb) ? y + 1 : 0;
                                    xy_reduction_int = 0;
                                    dist_xy = distance_xy_block[y + x * block_size];
                                    // contains_tie = 0.f;
                                    // loop_len = block_size - zstart;
                                    if(ybl2 == zbl2 && yb == zb){
                                        loop_len = block_size - y - 1;
                                        #pragma unroll(16)
                                        for (z = 0; z < block_size - y - 1; ++z){
                                            //compute masks for conflict blocks.

                                            distance_check_1 = dist_xy < distance_xz_block[z + y + 1 + x * block_size];
                                            distance_check_2 = dist_xy < distance_yz_block[z + y + 1 + y * block_size];
                                            scalar_xy_closest_int = distance_check_1 & distance_check_2;

                                            distance_check_1 = distance_xz_block[z + y + 1 + x * block_size] < dist_xy;
                                            distance_check_2 = distance_xz_block[z + y + 1 + x * block_size] < distance_yz_block[z + y + 1 + y * block_size];
                                            scalar_xz_closest_int = distance_check_1 & distance_check_2;

                                            distance_check_1 = distance_yz_block[z + y + 1 + y * block_size] < distance_xz_block[z + y + 1 + x * block_size];
                                            distance_check_2 = distance_yz_block[z + y + 1 + y * block_size] < dist_xy;
                                            scalar_yz_closest_int = distance_check_1 & distance_check_2;

                                            xy_reduction_int += scalar_xz_closest_int + scalar_yz_closest_int;
                                            buffer_conflict_yz_block_int[z + y + 1 + y * block_size] += scalar_xy_closest_int + scalar_xz_closest_int;
                                            buffer_conflict_xz_block_int[z + y + 1 + x * block_size] += scalar_xy_closest_int + scalar_yz_closest_int;
                                            // conflict_yz_ptr[z + y + 1 + y * block_size] += scalar_xy_closest_int + scalar_xz_closest_int;
                                            // conflict_xz_ptr[z + y + 1 + x * block_size] += scalar_xy_closest_int + scalar_yz_closest_int;
                                        }
                                    }
                                    else{
                                        #pragma unroll(16)
                                        for (z = 0; z < block_size; ++z){
                                            //compute masks for conflict blocks.
                                            distance_check_1 = dist_xy < distance_xz_block[z + x * block_size];
                                            distance_check_2 = dist_xy < distance_yz_block[z + y * block_size];
                                            scalar_xy_closest_int = distance_check_1 & distance_check_2;

                                            distance_check_1 = distance_xz_block[z + x * block_size] < dist_xy;
                                            distance_check_2 = distance_xz_block[z + x * block_size] < distance_yz_block[z + y * block_size];
                                            scalar_xz_closest_int = distance_check_1 & distance_check_2;

                                            distance_check_1 = distance_yz_block[z + y * block_size] < distance_xz_block[z + x * block_size];
                                            distance_check_2 = distance_yz_block[z + y * block_size] < dist_xy;
                                            scalar_yz_closest_int = distance_check_1 & distance_check_2;

                                            xy_reduction_int += scalar_xz_closest_int + scalar_yz_closest_int;
                                            buffer_conflict_yz_block_int[z + y * block_size] += scalar_xy_closest_int + scalar_xz_closest_int;
                                            buffer_conflict_xz_block_int[z + x * block_size] += scalar_xy_closest_int + scalar_yz_closest_int;
                                            // conflict_yz_ptr[z + y * block_size] += scalar_xy_closest_int + scalar_xz_closest_int;
                                            // conflict_xz_ptr[z + x * block_size] += scalar_xy_closest_int + scalar_yz_closest_int;

                                        }
                                    }
                                    buffer_conflict_xy_block_int[y + x * block_size] += xy_reduction_int;
                                    // conflict_xy_ptr[y + x * block_size] += xy_reduction_int;
                                }

                            }
                            conflict_loop_time += omp_get_wtime() - time_start;
                            time_start2 = omp_get_wtime();
                            conflict_xz_block_int = conflict_matrix_int + zb + zbl2 + xb * n + xbl2 * n;
                            conflict_yz_block_int = conflict_matrix_int + zb + zbl2 + yb * n + ybl2 * n;
                            for(i = 0; i < block_size; ++i){
                                for(j = 0; j < block_size; ++j){
                                    conflict_xz_block_int[j + i * n] += buffer_conflict_xz_block_int[j + i * block_size];
                                    conflict_yz_block_int[j + i * n] += buffer_conflict_yz_block_int[j + i * block_size];
                                }

                            }
                            memops_loop_time += omp_get_wtime() - time_start2;

                        }
                        time_start2 = omp_get_wtime();
                        conflict_xy_block_int = conflict_matrix_int + yb + ybl2 + xb * n + xbl2 * n;
                        for(i = 0; i < block_size; ++i){
                            for(j = 0; j < block_size; ++j){
                                conflict_xy_block_int[j + i * n] += buffer_conflict_xy_block_int[j + i * block_size];
                            }
                            // conflict_xy_block += n;
                        }
                        memops_loop_time += omp_get_wtime() - time_start2;
                    }
                }
                time_start2 = omp_get_wtime();
                // write UXZ and UYZ L2 blocks.
                // conflict_xz_block_int = conflict_matrix_int + zbl2 + xbl2 * n;
                // conflict_yz_block_int = conflict_matrix_int + zbl2 + ybl2 * n;
                // for(i = 0; i < l2_block_size; ++i){
                //     for(j = 0; j < l2_block_size; ++j){
                //         conflict_xz_block_int[j + i * n] += buffer_conflict_xz_L2_block_int[j + i * l2_block_size];
                //         conflict_yz_block_int[j + i * n] += buffer_conflict_yz_L2_block_int[j + i * l2_block_size];
                //     }
                // }
                // memops_loop_time += omp_get_wtime() - time_start2;
            }
            // time_start2 = omp_get_wtime();
            // // write UXY L2 block.
            // conflict_xy_block_int = conflict_matrix_int + ybl2 + xbl2 * n;
            // for(i = 0; i < l2_block_size; ++i){
            //     for(j = 0; j < l2_block_size; ++j){
            //         conflict_xy_block_int[j + i * n] += buffer_conflict_xy_L2_block_int[j + i * l2_block_size];
            //     }
            // }
            // memops_loop_time += omp_get_wtime() - time_start2;
        }
    }

    float* restrict conflict_matrix = (float *)  _mm_malloc(n * n * sizeof(float), VECALIGN);
    memset(conflict_matrix, 0, n * n * sizeof(float));
    // print_matrix_int(n, n, conflict_matrix_int);
    time_start = omp_get_wtime();
    for(i = 0; i < n * n; ++i){
        conflict_matrix[i] = 1.f/conflict_matrix_int[i];
    }
    // print_matrix(n, n, conflict_matrix);
    conflict_loop_time += omp_get_wtime() - time_start;
    _mm_free(conflict_matrix_int);
    _mm_free(buffer_conflict_xy_block_int);
    _mm_free(buffer_conflict_xz_block_int);
    _mm_free(buffer_conflict_yz_block_int);
    _mm_free(distance_xy_block);
    _mm_free(distance_xz_block);
    _mm_free(distance_yz_block);

    // initialize diagonal of C.
    float sum;
    time_start = omp_get_wtime();
    for (i = 0; i < n; ++i){
        sum = 0.f;
        for (j = 0; j < i; ++j){
            sum += conflict_matrix[i + j * n];
        }

        for (j = i + 1; j < n; ++j){
            sum += conflict_matrix[j + i * n];
        }
        C[i + i * n] = sum;
    }
    cohesion_loop_time += omp_get_wtime() - time_start;
    block_size /= 2;
    // l2_block_size /= 2;
    float scalar_xy_closest, scalar_xz_closest, scalar_yz_closest;
    float* restrict buffer_zx_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
    float* restrict buffer_zy_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
    float* restrict buffer_xz_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
    float* restrict buffer_yz_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
    float* restrict buffer_xy_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
    float* restrict buffer_yx_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);

    // float* restrict buffer_zx_L2_block = (float *) _mm_malloc(l2_block_size * l2_block_size * sizeof(float), VECALIGN);
    // float* restrict buffer_zy_L2_block = (float *) _mm_malloc(l2_block_size * l2_block_size * sizeof(float), VECALIGN);
    // float* restrict buffer_xz_L2_block = (float *) _mm_malloc(l2_block_size * l2_block_size * sizeof(float), VECALIGN);
    // float* restrict buffer_yz_L2_block = (float *) _mm_malloc(l2_block_size * l2_block_size * sizeof(float), VECALIGN);
    // float* restrict buffer_xy_L2_block = (float *) _mm_malloc(l2_block_size * l2_block_size * sizeof(float), VECALIGN);
    // float* restrict buffer_yx_L2_block = (float *) _mm_malloc(l2_block_size * l2_block_size * sizeof(float), VECALIGN);


    float* restrict buffer_conflict_xz_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
    float* restrict buffer_conflict_yz_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
    float* restrict buffer_conflict_xy_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
    distance_xy_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
    distance_xz_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
    distance_yz_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);

    // float * restrict conflict_xz_L2_block = (float *) _mm_malloc(l2_block_size * l2_block_size * sizeof(float), VECALIGN);
    // float * restrict conflict_yz_L2_block = (float *) _mm_malloc(l2_block_size * l2_block_size * sizeof(float), VECALIGN);
    // float * restrict conflict_xy_L2_block = (float *) _mm_malloc(l2_block_size * l2_block_size * sizeof(float), VECALIGN);

    // float *conflict_xy_block, *conflict_xz_block, *conflict_yz_block;
    float *cohesion_xy_block;
    float *cohesion_yx_block;
    float *cohesion_xz_block;
    float *cohesion_zx_block;
    float *cohesion_yz_block;
    float *cohesion_zy_block;
    float xy_reduction = 0.f, yx_reduction = 0.f;

    // ANNOTATE_SITE_BEGIN("Coh");
    for(xbl2 = 0; xbl2 < n; xbl2 += l2_block_size){
        for(ybl2 = xbl2; ybl2 < n; ybl2 += l2_block_size){
            for(zbl2 = ybl2; zbl2 < n; zbl2 += l2_block_size){
            // for(i = 0; i < l2_block_size; ++i){
            //     memcpy(distance_xy_L2_block + i * l2_block_size, D + ybl2 + (xbl2 + i) * n, sizeof(float)*l2_block_size);
            //     memcpy(conflict_xy_L2_block + i * l2_block_size, conflict_matrix + ybl2 + (xbl2 + i) * n, sizeof(float)*l2_block_size);
            // }
            // memset(buffer_xy_L2_block, 0, sizeof(float)*l2_block_size*l2_block_size);
            // memset(buffer_yx_L2_block, 0, sizeof(float)*l2_block_size*l2_block_size);
            //     for (i = 0; i < l2_block_size; ++i){
            //         memcpy(distance_xz_L2_block + i * l2_block_size, D + zbl2 + (xbl2 + i) * n, sizeof(float)*l2_block_size);
            //         memcpy(conflict_xz_L2_block + i * l2_block_size, conflict_matrix + zbl2 + (xbl2 + i) * n, sizeof(float)*l2_block_size);
            //     }
            //     for (i = 0; i < l2_block_size; ++i){
            //         memcpy(distance_yz_L2_block + i * l2_block_size, D + zbl2 + (ybl2 + i) * n, sizeof(float)*l2_block_size);
            //         memcpy(conflict_yz_L2_block + i * l2_block_size, conflict_matrix + zbl2 + (ybl2 + i) * n, sizeof(float)*l2_block_size);
            //     }
            //     memset(buffer_xz_L2_block, 0, sizeof(float)*l2_block_size*l2_block_size);
            //     memset(buffer_zx_L2_block, 0, sizeof(float)*l2_block_size*l2_block_size);
            //     memset(buffer_zy_L2_block, 0, sizeof(float)*l2_block_size*l2_block_size);
            //     memset(buffer_yz_L2_block, 0, sizeof(float)*l2_block_size*l2_block_size);
                for(xb = 0; xb < l2_block_size; xb += block_size){
                    ybstart = (xbl2 == ybl2) ? (xb) : 0;
                    for(yb = ybstart; yb < l2_block_size; yb += block_size){
                        time_start = omp_get_wtime();

                        // distance_xy_block = distance_xy_L2_block + yb + xb * l2_block_size;
                        // conflict_xy_block = conflict_xy_L2_block + yb + xb * l2_block_size;
                        for(i = 0; i < block_size; ++i){
                            memcpy(buffer_conflict_xy_block + i * block_size, conflict_matrix + yb + ybl2 + (xbl2 + xb + i) * n, sizeof(float)*block_size);
                            memcpy(distance_xy_block + i * block_size, D + yb + ybl2 + (xbl2 + xb + i) * n, sizeof(float)*block_size);
                        }

                        memset(buffer_yx_block, 0, sizeof(float)*block_size*block_size);
                        memset(buffer_xy_block, 0, sizeof(float)*block_size*block_size);
                        // buffer_xy_block = buffer_xy_L2_block + yb + xb * l2_block_size;
                        // buffer_yx_block = buffer_yx_L2_block + yb + xb * l2_block_size;
                        memops_loop_time += omp_get_wtime() - time_start;
                        zbstart = (zbl2 == ybl2) ? (yb) : 0;
                        // printf("zb start\n");
                        for(zb = zbstart; zb < l2_block_size; zb += block_size){
                           time_start = omp_get_wtime();
                        //    distance_xz_block = distance_xz_L2_block + zb + xb * l2_block_size;
                        //    conflict_xz_block = conflict_xz_L2_block + zb + xb * l2_block_size;

                        //    distance_yz_block = distance_xz_L2_block + zb + yb * l2_block_size;
                        //    conflict_yz_block = conflict_yz_L2_block + zb + yb * l2_block_size;

                            for (i = 0; i < block_size; ++i){
                                memcpy(buffer_conflict_xz_block + i * block_size, conflict_matrix + zbl2 + zb + (xbl2 + xb + i) * n, sizeof(float)*block_size);
                                memcpy(distance_xz_block + i * block_size, D  + zbl2 + zb + (xbl2 + xb + i) * n, sizeof(float)*block_size);
                            }
                            for (i = 0; i < block_size; ++i){
                                memcpy(buffer_conflict_yz_block + i * block_size, conflict_matrix + zbl2 + zb + (ybl2 + yb + i) * n, sizeof(float)*block_size);
                                memcpy(distance_yz_block + i * block_size, D + zbl2 + zb + (ybl2 + yb + i) * n, sizeof(float)*block_size);
                            }
                            memset(buffer_zx_block, 0, sizeof(float)*block_size*block_size);
                            memset(buffer_zy_block, 0, sizeof(float)*block_size*block_size);
                            memset(buffer_xz_block, 0, sizeof(float)*block_size*block_size);
                            memset(buffer_yz_block, 0, sizeof(float)*block_size*block_size);
                            // buffer_xz_block = buffer_xz_L2_block + zb + xb * l2_block_size;
                            // buffer_zx_block = buffer_zx_L2_block + zb + xb * l2_block_size;

                            // buffer_yz_block = buffer_yz_L2_block + zb + yb * l2_block_size;
                            // buffer_zy_block = buffer_zy_L2_block + zb + yb * l2_block_size;
                            memops_loop_time += omp_get_wtime() - time_start;

                            time_start = omp_get_wtime();
                            xend = block_size;


                            if(xbl2 == ybl2 && ybl2 == zbl2 && xb == yb && yb == zb){
                                xend = block_size - 1;
                            }

                            for(x = 0; x < xend; ++x){
                                ystart = (xbl2 == ybl2 && xb == yb) ? x + 1 : 0;
                                for(y = ystart; y < block_size; ++y){
                                    xy_reduction = 0.f; yx_reduction = 0.f;
                                    // if(ybl2 == zbl2 && yb == zb){
                                    //     zstart = y + 1;
                                    // }
                                    // zstart = (ybl2 == zbl2 && yb == zb) ? y + 1 : 0;
                                    dist_xy = distance_xy_block[y + x * block_size];
                                    conflict_xy_val = buffer_conflict_xy_block[y + x * block_size];
                                    // loop_len = block_size - zstart;
                                    // contains_tie = 0.f;
                                    if(ybl2 == zbl2 && yb == zb){
                                        loop_len = block_size - y - 1;
                                        // conflict_xy_val = buffer_conflict_xy_block[y + x * block_size];
                                        idx = y + 1;
                                        #pragma unroll(8)
                                        for (z = 0; z < loop_len; ++z){
                                            //compute masks for conflict blocks.
                                            distance_check_1 = dist_xy < distance_xz_block[idx + x * block_size];
                                            distance_check_2 = dist_xy < distance_yz_block[idx + y * block_size];
                                            // distance_check_3 = distance_xz_block[idx + x * block_size] < distance_yz_block[idx + y * block_size];
                                            scalar_xy_closest = distance_check_1 & distance_check_2;
                                            yx_reduction += scalar_xy_closest*buffer_conflict_xz_block[idx + x * block_size];
                                            xy_reduction += scalar_xy_closest*buffer_conflict_yz_block[idx + y * block_size];
                                            // yx_reduction += scalar_xy_closest*conflict_xz_block[idx + x * block_size];
                                            // xy_reduction += scalar_xy_closest*conflict_yz_block[idx + y * block_size];

                                            distance_check_1 = distance_xz_block[idx + x * block_size] < dist_xy;
                                            distance_check_2 = distance_xz_block[idx + x * block_size] < distance_yz_block[idx + y * block_size];
                                            scalar_xz_closest = distance_check_1 & distance_check_2;
                                            buffer_xz_block[idx + x * block_size] += scalar_xz_closest*buffer_conflict_yz_block[idx + y * block_size];
                                            buffer_zx_block[idx + x * block_size] += scalar_xz_closest*conflict_xy_val;
                                            // buffer_xz_block[idx + x * block_size] += scalar_xz_closest*conflict_yz_block[idx + y * block_size];
                                            // buffer_zx_block[idx + x * block_size] += scalar_xz_closest*conflict_xy_val;

                                            distance_check_1 = distance_yz_block[idx + y * block_size] < distance_xz_block[idx + x * block_size];
                                            distance_check_2 = distance_yz_block[idx + y * block_size] < dist_xy;
                                            scalar_yz_closest = distance_check_1 & distance_check_2;
                                            buffer_yz_block[idx + y * block_size] += scalar_yz_closest*buffer_conflict_xz_block[idx + x * block_size];
                                            buffer_zy_block[idx + y * block_size] += scalar_yz_closest*conflict_xy_val;
                                            idx++;
                                            // buffer_yz_block[idx + y * block_size] += scalar_yz_closest*conflict_xz_block[idx + x * block_size];
                                            // buffer_zy_block[idx + y * block_size] += scalar_yz_closest*conflict_xy_val;
                                        }
                                        buffer_xy_block[y + x * block_size] += xy_reduction;
                                        buffer_yx_block[y + x * block_size] += yx_reduction;

                                    }
                                    else{
                                        // conflict_xy_val = buffer_conflict_xy_block[y + x * block_size];
                                        #pragma unroll(8)
                                        for (z = 0; z < block_size; ++z){
                                            //compute masks for conflict blocks.
                                            distance_check_1 = dist_xy < distance_xz_block[z + x * block_size];
                                            distance_check_2 = dist_xy < distance_yz_block[z + y * block_size];
                                            // distance_check_3 = distance_xz_block[idx + x * block_size] < distance_yz_block[idx + y * block_size];
                                            scalar_xy_closest = distance_check_1 & distance_check_2;
                                            yx_reduction += scalar_xy_closest*buffer_conflict_xz_block[z + x * block_size];
                                            xy_reduction += scalar_xy_closest*buffer_conflict_yz_block[z + y * block_size];
                                            // xy_reduction += scalar_xy_closest*conflict_yz_block[z + y * block_size];
                                            // yx_reduction += scalar_xy_closest*conflict_xz_block[z + x * block_size];

                                            distance_check_1 = distance_xz_block[z + x * block_size] < dist_xy;
                                            distance_check_2 = distance_xz_block[z + x * block_size] < distance_yz_block[z + y * block_size];
                                            scalar_xz_closest = distance_check_1 & distance_check_2;
                                            buffer_xz_block[z + x * block_size] += scalar_xz_closest*buffer_conflict_yz_block[z + y * block_size];
                                            buffer_zx_block[z + x * block_size] += scalar_xz_closest*conflict_xy_val;
                                            // buffer_xz_block[z + x * block_size] += scalar_xz_closest*conflict_yz_block[z + y * block_size];
                                            // buffer_zx_block[z + x * block_size] += scalar_xz_closest*conflict_xy_val;

                                            distance_check_1 = distance_yz_block[z + y * block_size] < distance_xz_block[z + x * block_size];
                                            distance_check_2 = distance_yz_block[z + y * block_size] < dist_xy;
                                            scalar_yz_closest = distance_check_1 & distance_check_2;
                                            buffer_yz_block[z + y * block_size] += scalar_yz_closest*buffer_conflict_xz_block[z + x * block_size];
                                            buffer_zy_block[z + y * block_size] += scalar_yz_closest*conflict_xy_val;
                                            // buffer_yz_block[z + y * block_size] += scalar_yz_closest*conflict_xz_block[z + x * block_size];
                                            // buffer_zy_block[z + y * block_size] += scalar_yz_closest*conflict_xy_val;
                                        }

                                        buffer_xy_block[y + x * block_size] += xy_reduction;
                                        buffer_yx_block[y + x * block_size] += yx_reduction;
                                    }
                                }
                            }
                            cohesion_loop_time += omp_get_wtime() - time_start;
                            cohesion_zx_block = C + xb + xbl2 + zbl2 * n + zb * n;
                            cohesion_zy_block = C + yb + ybl2 + zbl2 * n + zb * n;
                            cohesion_xz_block = C + zb + zbl2 + xbl2 * n + xb * n;
                            cohesion_yz_block = C + zb + zbl2 + ybl2 * n + yb * n;
                            // cohesion_xy_block = C + yb + xb * n;
                            // cohesion_yx_block = C + xb + yb * n;
                            time_start2 = omp_get_wtime();
                            for(i = 0; i < block_size; ++i){
                                for(j = 0; j < block_size; ++j){
                                    cohesion_zx_block[j + i * n] += buffer_zx_block[i + j * block_size];
                                    cohesion_zy_block[j + i * n] += buffer_zy_block[i + j * block_size];
                                }
                                // cohesion_zx_block += n;
                                // cohesion_zy_block += n;
                            }

                            for(i = 0; i < block_size; ++i){
                                for(j = 0; j < block_size; ++j){
                                    cohesion_xz_block[j + i * n] += buffer_xz_block[j + i * block_size];
                                    cohesion_yz_block[j + i * n] += buffer_yz_block[j + i * block_size];
                                }
                                // cohesion_xz_block += n;
                                // cohesion_yz_block += n;
                            }

                            memops_loop_time += omp_get_wtime() - time_start2;
                        }
                        // printf("zb end\n");
                        time_start2 = omp_get_wtime();
                        cohesion_xy_block = C + ybl2 + yb + xbl2 * n + xb * n;
                        cohesion_yx_block = C + xb + xbl2 + ybl2 * n + yb * n;
                        for(i = 0; i < block_size; ++i){
                            for(j = 0; j < block_size; ++j){
                                cohesion_yx_block[j + i * n] += buffer_yx_block[i + j * block_size];
                            }
                            // cohesion_yx_block += n;
                        }
                        cohesion_yx_block = C + xb + xbl2 + ybl2 * n + yb * n;

                        for(i = 0; i < block_size; ++i){
                            for(j = 0; j < block_size; ++j){
                                cohesion_xy_block[j + i * n] += buffer_xy_block[j + i * block_size];
                            }
                            // cohesion_xy_block += n;
                        }
                        memops_loop_time += omp_get_wtime() - time_start2;
                    }
                }
                // time_start2 = omp_get_wtime();
                // cohesion_xz_block = C + zbl2 + xbl2 * n;
                // cohesion_zx_block = C + xbl2 + zbl2 * n;

                // cohesion_yz_block = C + zbl2 + ybl2 * n;
                // cohesion_zy_block = C + ybl2 + zbl2 * n;

                // // cohesion_xy_block = C + yb + xb * n;
                // // cohesion_yx_block = C + xb + yb * n;
                // time_start2 = omp_get_wtime();
                // for(i = 0; i < l2_block_size; ++i){
                //     for(j = 0; j < l2_block_size; ++j){
                //         cohesion_zx_block[j] += buffer_zx_block[i + j * block_size];
                //         cohesion_zy_block[j] += buffer_zy_block[i + j * block_size];
                //     }
                //     cohesion_zx_block += n;
                //     cohesion_zy_block += n;
                // }

                // for(i = 0; i < l2_block_size; ++i){
                //     for(j = 0; j < l2_block_size; ++j){
                //         cohesion_xz_block[j] += buffer_xz_block[j + i * block_size];
                //         cohesion_yz_block[j] += buffer_yz_block[j + i * block_size];
                //     }
                //     cohesion_xz_block += n;
                //     cohesion_yz_block += n;
                // }

                // memops_loop_time += omp_get_wtime() - time_start2;

            }

            // time_start2 = omp_get_wtime();
            // cohesion_xy_block = C + ybl2 + xbl2 * n;
            // cohesion_yx_block = C + xbl2 + ybl2 * n;
            // for(i = 0; i < l2_block_size; ++i){
            //     for(j = 0; j < l2_block_size; ++j){
            //         cohesion_yx_block[j] += buffer_yx_block[i + j * block_size];
            //     }
            //     cohesion_yx_block += n;
            // }
            // for(i = 0; i < l2_block_size; ++i){
            //     for(j = 0; j < l2_block_size; ++j){
            //         cohesion_xy_block[j] += buffer_xy_block[j + i * block_size];
            //     }
            //     cohesion_xy_block += n;
            // }
            // memops_loop_time += omp_get_wtime() - time_start2;
        }
    }
    // print_matrix(n, n, C);
    // ANNOTATE_SITE_END("Coh");
    // print_matrix(n, n, C);

    printf("======================================\n");
    printf("Seq. Triplet Multi-level Blocking Times\n");
    printf("======================================\n");

    printf("memops loop time: %.5fs\n", memops_loop_time);
    printf("conflict focus size loop time: %.5fs\n", conflict_loop_time);
    printf("cohesion matrix update loop time: %.5fs\n\n", cohesion_loop_time);


    _mm_free(conflict_matrix);
    _mm_free(distance_xy_block); _mm_free(distance_xz_block); _mm_free(distance_yz_block);
    _mm_free(buffer_zx_block); _mm_free(buffer_zy_block); _mm_free(buffer_yx_block);
    _mm_free(buffer_xz_block); _mm_free(buffer_yz_block); _mm_free(buffer_xy_block);
    _mm_free(buffer_conflict_xz_block); _mm_free(buffer_conflict_yz_block); _mm_free(buffer_conflict_xy_block);

    // _mm_free(conflict_L2_block); _mm_free(conflict_yz_L2_block); _mm_free(conflict_xy_L2_block);

    // _mm_free(buffer_conflict_xy_L2_block); _mm_free(buffer_conflict_xz_L2_block); _mm_free(buffer_conflict_yz_L2_block);

    // _mm_free(distance_xy_block);
    // _mm_free(distance_xz_block); _mm_free(distance_yz_block);
    // _mm_free(mask_tie_xy_xz); _mm_free(mask_tie_xy_yz); _mm_free(mask_tie_xz_yz);
    // _mm_free(mask_xy_closest); _mm_free(mask_xz_closest); _mm_free(mask_yz_closest);
    // _mm_free(buffer_zx_block); _mm_free(buffer_zy_block); _mm_free(buffer_yx_block);
    // _mm_free(buffer_xz_block); _mm_free(buffer_yz_block); _mm_free(buffer_xy_block);

}


void pald_triplet_openmp_powersoftwo(float *D, float beta, int n, float *C, int block_size, int nthreads){
    //TODO: Optimized sequential triplet code.
    float* restrict conflict_matrix = (float *)  _mm_malloc(n * n * sizeof(float), VECALIGN);
    // memset(conflict_matrix, 0, n * n * sizeof(float));
    unsigned int* restrict conflict_matrix_int = (unsigned int*)  _mm_malloc(n * n * sizeof(unsigned int), VECALIGN);
    double time_start = 0.0, conflict_loop_time = 0.0, cohesion_loop_time;
    // memset(conflict_matrix_int, 0, n * n * sizeof(unsigned int));
    time_start = omp_get_wtime();
    #pragma omp parallel shared(D, conflict_matrix_int, conflict_matrix, n) num_threads(nthreads)
    {
        // printf("nthreads: %d\n", omp_get_num_threads());
        unsigned int scalar_xy_closest_int, scalar_xz_closest_int, scalar_yz_closest_int;
        unsigned int distance_check_1_mask, distance_check_2_mask;
        unsigned int xy_reduction_int;

        unsigned int *conflict_xy_int, *conflict_xz_int, *conflict_yz_int;
        float dist_xy  = 0.f;
        float conflict_xy_val = 0.f;
        unsigned int loop_len = 0;

        // char print_out = 0;
        double time_start = 0.0, time_start2 = 0.0;
        int xb, yb, zb, x, y, z;
        int i, j, k;
        // int size_xy = block_size, size_xz = block_size, size_yz = block_size;
        int xend, ystart, zstart;
        float xy_reduction = 0.f, yx_reduction = 0.f, cohesion_sum = 0.f;
        // compute conflict focus sizes.
        int iters = 0;
        int idx;
        #pragma omp for
        for (i = 0; i < n; ++i){
            for (j = i + 1; j < n; ++j){
                // conflict_matrix[j + i * n] = 2.;
                conflict_matrix_int[j + i * n] = 2;
            }
        }
        // print_matrix_int(n, n, conflict_matrix_int);
        // if(print_out)
        //     print_matrix(n, n, conflict_matrix);

        #pragma omp single nowait
        for(xb = 0; xb < n; xb += block_size){
            for(yb = xb; yb < n; yb += block_size){
                for(zb = yb; zb < n; zb += block_size){
                    // Set DXZ and DYZ blocks.
                    #pragma omp task untied \
                    shared(n, block_size, D, conflict_matrix_int) \
                    firstprivate(xb, yb, zb) \
                    depend(inout: conflict_matrix[yb + xb * n]) \
                    depend(inout: conflict_matrix[zb + xb * n]) \
                    depend(inout: conflict_matrix[zb + yb * n])
                    {
                        float* restrict distance_xy_block = (float*) _mm_malloc(sizeof(float) * block_size * block_size, VECALIGN);
                        float* restrict distance_xz_block = (float*) _mm_malloc(sizeof(float) * block_size * block_size, VECALIGN);
                        float* restrict distance_yz_block = (float*) _mm_malloc(sizeof(float) * block_size * block_size, VECALIGN);
                        unsigned int* restrict conflict_xy_block_int = (unsigned int *) _mm_malloc(sizeof(int) * block_size * block_size, VECALIGN);
                        unsigned int* restrict conflict_xz_block_int = (unsigned int *) _mm_malloc(sizeof(int) * block_size * block_size, VECALIGN);
                        unsigned int* restrict conflict_yz_block_int = (unsigned int *) _mm_malloc(sizeof(int) * block_size * block_size, VECALIGN);

                        // Copy DXY, DXZ, DYZ blocks.
                        for(i = 0; i < block_size; ++i){
                            memcpy(distance_xy_block + i * block_size, D + yb + (xb + i) * n, sizeof(float) * block_size);
                            memcpy(distance_xz_block + i * block_size, D + zb + (xb + i) * n, sizeof(float) * block_size);
                            memcpy(distance_yz_block + i * block_size, D + zb + (yb + i) * n, sizeof(float) * block_size);
                        }
                        // Set UXY, UXZ, and UYZ blocks.
                        memset(conflict_xy_block_int, 0, sizeof(int) * block_size * block_size);
                        memset(conflict_xz_block_int, 0, sizeof(int) * block_size * block_size);
                        memset(conflict_yz_block_int, 0, sizeof(int) * block_size * block_size);
                        xend = (xb == yb && yb == zb) ? block_size - 1 : block_size;
                        for(x = 0; x < xend; ++x){
                            ystart = (xb == yb) ? x + 1 : 0;
                            if(xb == yb){
                                for(y = x + 1; y < block_size; ++y){
                                    // if(yb == zb){
                                    //     zstart = y + 1;
                                    // }
                                    // xy_reduction = 0.f;
                                    xy_reduction_int = 0;
                                    zstart = (yb == zb) ? y + 1 : 0;
                                    dist_xy = distance_xy_block[y + x * block_size];
                                    // contains_tie = 0.f;
                                    loop_len = block_size - zstart;
                                    if(yb == zb){
                                        // for (z = y + 1; z < block_size; ++z){
                                        #pragma unroll(16)
                                        for (z = 0; z < loop_len; ++z){
                                            idx = z + y + 1;
                                            //compute masks for conflict blocks.
                                            distance_check_1_mask = dist_xy < distance_xz_block[idx + x * block_size];
                                            distance_check_2_mask = dist_xy < distance_yz_block[idx + y * block_size];
                                            scalar_xy_closest_int = distance_check_1_mask & distance_check_2_mask;

                                            distance_check_1_mask = distance_xz_block[idx + x * block_size] < dist_xy;
                                            distance_check_2_mask =  distance_xz_block[idx + x * block_size] < distance_yz_block[idx + y * block_size];
                                            scalar_xz_closest_int = distance_check_1_mask & distance_check_2_mask;

                                            distance_check_1_mask = distance_yz_block[idx + y * block_size] < distance_xz_block[idx + x * block_size];
                                            distance_check_2_mask = distance_yz_block[idx + y * block_size] < dist_xy;
                                            scalar_yz_closest_int = distance_check_1_mask & distance_check_2_mask;


                                            xy_reduction_int += scalar_xz_closest_int + scalar_yz_closest_int;
                                            conflict_yz_block_int[idx + y * block_size] += scalar_xy_closest_int + scalar_xz_closest_int;
                                            conflict_xz_block_int[idx + x * block_size] += scalar_xy_closest_int + scalar_yz_closest_int;
                                        }
                                        conflict_xy_block_int[y + x * block_size] += xy_reduction_int;
                                    }
                                    else{
                                        #pragma unroll(16)
                                        for (z = 0; z < block_size; ++z){
                                            //compute masks for conflict blocks.
                                            distance_check_1_mask = dist_xy < distance_xz_block[z + x * block_size];
                                            distance_check_2_mask = dist_xy < distance_yz_block[z + y * block_size];
                                            scalar_xy_closest_int = distance_check_1_mask & distance_check_2_mask;

                                            distance_check_1_mask = distance_xz_block[z + x * block_size] < dist_xy;
                                            distance_check_2_mask =  distance_xz_block[z + x * block_size] < distance_yz_block[z + y * block_size];
                                            scalar_xz_closest_int = distance_check_1_mask & distance_check_2_mask;

                                            distance_check_1_mask = distance_yz_block[z + y * block_size] < distance_xz_block[z + x * block_size];
                                            distance_check_2_mask = distance_yz_block[z + y * block_size] < dist_xy;
                                            scalar_yz_closest_int = distance_check_1_mask & distance_check_2_mask;

                                            xy_reduction_int += scalar_xz_closest_int + scalar_yz_closest_int;
                                            conflict_yz_block_int[z + y * block_size] += scalar_xy_closest_int + scalar_xz_closest_int;
                                            conflict_xz_block_int[z + x * block_size] += scalar_xy_closest_int + scalar_yz_closest_int;
                                        }
                                        conflict_xy_block_int[y + x * block_size] += xy_reduction_int;
                                    }
                                }
                            }
                            else{
                                for(y = 0; y < block_size; ++y){
                                    xy_reduction_int = 0;
                                    zstart = (yb == zb) ? y + 1 : 0;
                                    dist_xy = distance_xy_block[y + x * block_size];
                                    // contains_tie = 0.f;
                                    loop_len = block_size - zstart;
                                    if(yb == zb){
                                        #pragma unroll(16)
                                        for (z = 0; z < loop_len; ++z){
                                            idx = z + y + 1;
                                            //compute masks for conflict blocks.
                                            distance_check_1_mask = dist_xy < distance_xz_block[idx + x * block_size];
                                            distance_check_2_mask = dist_xy < distance_yz_block[idx + y * block_size];
                                            scalar_xy_closest_int = distance_check_1_mask & distance_check_2_mask;

                                            distance_check_1_mask = distance_xz_block[idx + x * block_size] < dist_xy;
                                            distance_check_2_mask =  distance_xz_block[idx + x * block_size] < distance_yz_block[idx + y * block_size];
                                            scalar_xz_closest_int = distance_check_1_mask & distance_check_2_mask;

                                            distance_check_1_mask = distance_yz_block[idx + y * block_size] < distance_xz_block[idx + x * block_size];
                                            distance_check_2_mask = distance_yz_block[idx + y * block_size] < dist_xy;
                                            scalar_yz_closest_int = distance_check_1_mask & distance_check_2_mask;


                                            xy_reduction_int += scalar_xz_closest_int + scalar_yz_closest_int;
                                            conflict_yz_block_int[idx + y * block_size] += scalar_xy_closest_int + scalar_xz_closest_int;
                                            conflict_xz_block_int[idx + x * block_size] += scalar_xy_closest_int + scalar_yz_closest_int;
                                        }
                                        conflict_xy_block_int[y + x * block_size] += xy_reduction_int;
                                    }
                                    else{
                                        #pragma unroll(16)
                                        for (z = 0; z < block_size; ++z){
                                            //compute masks for conflict blocks.
                                            distance_check_1_mask = dist_xy < distance_xz_block[z + x * block_size];
                                            distance_check_2_mask = dist_xy < distance_yz_block[z + y * block_size];
                                            scalar_xy_closest_int = distance_check_1_mask & distance_check_2_mask;

                                            distance_check_1_mask = distance_xz_block[z + x * block_size] < dist_xy;
                                            distance_check_2_mask =  distance_xz_block[z + x * block_size] < distance_yz_block[z + y * block_size];
                                            scalar_xz_closest_int = distance_check_1_mask & distance_check_2_mask;

                                            distance_check_1_mask = distance_yz_block[z + y * block_size] < distance_xz_block[z + x * block_size];
                                            distance_check_2_mask = distance_yz_block[z + y * block_size] < dist_xy;
                                            scalar_yz_closest_int = distance_check_1_mask & distance_check_2_mask;
                                            xy_reduction_int += scalar_xz_closest_int + scalar_yz_closest_int;
                                            conflict_yz_block_int[z + y * block_size] += scalar_xy_closest_int + scalar_xz_closest_int;
                                            conflict_xz_block_int[z + x * block_size] += scalar_xy_closest_int + scalar_yz_closest_int;
                                        }
                                        conflict_xy_block_int[y + x * block_size] += xy_reduction_int;
                                    }
                                }
                            }
                        }
                        // conflict_xy_int = conflict_matrix_int + yb + xb * n;
                        // conflict_xz_int = conflict_matrix_int + zb + xb * n;
                        // conflict_yz_int = conflict_matrix_int + zb + yb * n;
                        for(i = 0; i < block_size; ++i){
                            for(j = 0; j < block_size; ++j){
                                idx = j + i * n;
                                conflict_matrix_int[yb + xb * n + idx] += conflict_xy_block_int[j + i * block_size];
                                conflict_matrix_int[zb + xb * n + idx] += conflict_xz_block_int[j + i * block_size];
                                conflict_matrix_int[zb + yb * n + idx] += conflict_yz_block_int[j + i * block_size];
                            }
                        }
                        // printf("(xb: %d, yb: %d, zb: %d)\n", xb, yb, zb);
                        // print_matrix_int(block_size, block_size, conflict_yz_block_int);
                        // printf("[\n");
                        // for(i = 0; i < block_size; ++i){
                        //     for(j = 0; j < block_size; ++j){
                        //         printf("%d ", conflict_xy_int[j + i * n]);
                        //     }
                        //     printf(";\n");
                        // }
                        // printf("];\n");
                        _mm_free(distance_xy_block); _mm_free(distance_xz_block); _mm_free(distance_yz_block);
                        _mm_free(conflict_xy_block_int); _mm_free(conflict_xz_block_int); _mm_free(conflict_yz_block_int);
                    }
                }
                // print_matrix_int(n,n, conflict_matrix_int);
            }
        }
        #pragma omp barrier
        // #pragma omp master
        // print_matrix_int(n, n, conflict_matrix_int);
        #pragma omp for
        for(i = 0; i < n * n; ++i){
            // conflict_matrix[i] = 1.f/conflict_matrix[i];
            conflict_matrix[i] = 1.f/conflict_matrix_int[i];
        }
        // #pragma omp master
        // print_matrix_int(n, n, conflict_matrix_int);
        // return;
        // printf("\n\n");
            // initialize diagonal of C.

    }
    // print_matrix(n, n, conflict_matrix);
    _mm_free(conflict_matrix_int);

    conflict_loop_time = omp_get_wtime() - time_start;
    time_start = omp_get_wtime();
    #pragma omp parallel shared(C, D, conflict_matrix, n) num_threads(nthreads)
    {
        // block_size /= 2;
        int iters = 0;
        double time_start = 0.0, time_start2 = 0.0;
        int xb, yb, zb;
        int i, j;
        float sum = 0.f;
        #pragma omp for
        for (i = 0; i < n; ++i){
            sum = 0.f;
            for (j = 0; j < i; ++j){
                sum += conflict_matrix[i + j * n];
            }
            for (j = i + 1; j < n; ++j){
                sum += conflict_matrix[j + i * n];
            }
            C[i + i * n] = sum;
        }

        float scalar_xy_closest, scalar_xz_closest, scalar_yz_closest;
        #pragma omp single nowait
        for(xb = 0; xb < n; xb += block_size){
            for(yb = xb; yb < n; yb += block_size){
                for(zb = yb; zb < n; zb += block_size){
                    #pragma omp task untied \
                    shared(n, block_size, D, conflict_matrix, C) \
                    firstprivate(xb, yb, zb) \
                    depend(inout: C[yb + xb * n]) \
                    depend(inout: C[xb + yb * n]) \
                    depend(inout: C[zb + xb * n]) \
                    depend(inout: C[xb + zb * n]) \
                    depend(inout: C[zb + yb * n]) \
                    depend(inout: C[yb + zb * n])
                    {
                        // printf("tid: %d, (xb: %d, yb:%d, zb:%d)\n", omp_get_thread_num(), xb/block_size, yb/block_size, zb/block_size);
                        int x, y, z, idx, zstart, ystart, xend, loop_len;
                        unsigned int distance_check_1_mask, distance_check_2_mask;
                        float dist_xy = 0.f, conflict_xy_val = 0.f;
                        float xy_reduction = 0.f, yx_reduction = 0.f, cohesion_sum = 0.f;


                        // distance matrix cache blocks.
                        float* restrict distance_xy_block = (float*) _mm_malloc(sizeof(float) * block_size * block_size, VECALIGN);
                        float* restrict distance_xz_block = (float*) _mm_malloc(sizeof(float) * block_size * block_size, VECALIGN);
                        float* restrict distance_yz_block = (float*) _mm_malloc(sizeof(float) * block_size * block_size, VECALIGN);
                        // conflict matrix cache blocks.
                        float* restrict conflict_xy_block = (float*) _mm_malloc(sizeof(float) * block_size * block_size, VECALIGN);
                        float* restrict conflict_xz_block = (float*) _mm_malloc(sizeof(float) * block_size * block_size, VECALIGN);
                        float* restrict conflict_yz_block = (float*) _mm_malloc(sizeof(float) * block_size * block_size, VECALIGN);
                        // cohesion matrix cache blocks.
                        float* restrict cohesion_xy_block = (float*) _mm_malloc(sizeof(float) * block_size * block_size, VECALIGN);
                        float* restrict cohesion_yx_block = (float*) _mm_malloc(sizeof(float) * block_size * block_size, VECALIGN);
                        float* restrict cohesion_xz_block = (float*) _mm_malloc(sizeof(float) * block_size * block_size, VECALIGN);
                        float* restrict cohesion_zx_block = (float*) _mm_malloc(sizeof(float) * block_size * block_size, VECALIGN);
                        float* restrict cohesion_yz_block = (float*) _mm_malloc(sizeof(float) * block_size * block_size, VECALIGN);
                        float* restrict cohesion_zy_block = (float*) _mm_malloc(sizeof(float) * block_size * block_size, VECALIGN);
                        // Set distance blocks: DXY, DXZ, DYZ.
                        for(i = 0; i < block_size; ++i){
                            memcpy(distance_xy_block + i * block_size, D + yb + (xb + i) * n, sizeof(float) * block_size);
                            memcpy(distance_xz_block + i * block_size, D + zb + (xb + i) * n, sizeof(float) * block_size);
                            memcpy(distance_yz_block + i * block_size, D + zb + (yb + i) * n, sizeof(float) * block_size);
                        }
                        // Set conflict blocks: UXY, UXZ, UYZ.
                        for(i = 0; i < block_size; ++i){
                            memcpy(conflict_xy_block + i * block_size, conflict_matrix + yb + (xb + i) * n, sizeof(float) * block_size);
                            memcpy(conflict_xz_block + i * block_size, conflict_matrix + zb + (xb + i) * n, sizeof(float) * block_size);
                            memcpy(conflict_yz_block + i * block_size, conflict_matrix + zb + (yb + i) * n, sizeof(float) * block_size);
                        }
                        // Set cohesion blocks: CXY, CYX.
                        memset(cohesion_xy_block, 0, sizeof(float) * block_size * block_size);
                        memset(cohesion_yx_block, 0, sizeof(float) * block_size * block_size);

                        // Set cohesion blocks: CXZ, CZX.
                        memset(cohesion_xz_block, 0, sizeof(float) * block_size * block_size);
                        memset(cohesion_zx_block, 0, sizeof(float) * block_size * block_size);

                        // Set cohesion blocks: CYZ, CZY.
                        memset(cohesion_yz_block, 0, sizeof(float) * block_size * block_size);
                        memset(cohesion_zy_block, 0, sizeof(float) * block_size * block_size);

                        xend = block_size;
                        // ystart = 0;
                        // zstart = 0;
                        if(xb == yb && yb == zb){
                            xend = block_size - 1;
                        }
                        for(x = 0; x < xend; ++x){
                            ystart = (xb == yb) ? x + 1 : 0;
                            for(y = ystart; y < block_size; ++y){
                                xy_reduction = 0.f; yx_reduction = 0.f;
                                dist_xy = distance_xy_block[y + x * block_size];
                                if(yb == zb){
                                    loop_len = block_size - y - 1;
                                    conflict_xy_val = conflict_xy_block[y + x * block_size];
                                    for (z = 0; z < loop_len; ++z){
                                        //compute masks for conflict blocks.
                                        idx = z + y + 1;
                                        distance_check_1_mask = dist_xy < distance_xz_block[idx + x * block_size];
                                        distance_check_2_mask = dist_xy < distance_yz_block[idx + y * block_size];
                                        scalar_xy_closest = distance_check_1_mask & distance_check_2_mask;

                                        distance_check_1_mask = distance_xz_block[idx + x * block_size] < dist_xy;
                                        distance_check_2_mask =  distance_xz_block[idx + x * block_size] < distance_yz_block[idx + y * block_size];
                                        scalar_xz_closest = distance_check_1_mask & distance_check_2_mask;

                                        distance_check_1_mask = distance_yz_block[idx + y * block_size] < distance_xz_block[idx + x * block_size];
                                        distance_check_2_mask = distance_yz_block[idx + y * block_size] < dist_xy;
                                        scalar_yz_closest = distance_check_1_mask & distance_check_2_mask;

                                        // xy closest pair.
                                        yx_reduction += scalar_xy_closest*conflict_xz_block[idx + x * block_size];
                                        xy_reduction += scalar_xy_closest*conflict_yz_block[idx + y * block_size];

                                        // xz closest pair.
                                        cohesion_xz_block[idx + x * block_size] += scalar_xz_closest*conflict_yz_block[idx + y * block_size];
                                        cohesion_zx_block[idx + x * block_size] += scalar_xz_closest*conflict_xy_val;

                                        // yz closest pair.
                                        cohesion_yz_block[idx + y * block_size] += scalar_yz_closest*conflict_xz_block[idx + x * block_size];
                                        cohesion_zy_block[idx + y * block_size] += scalar_yz_closest*conflict_xy_val;
                                    }
                                    cohesion_xy_block[y + x * block_size] += xy_reduction;
                                    cohesion_yx_block[y + x * block_size] += yx_reduction;
                                }
                                else{
                                    conflict_xy_val = conflict_xy_block[y + x * block_size];
                                    //update cohesion blocks.
                                    #pragma unroll(8)
                                    for (z = 0; z < block_size; ++z){
                                        // xy closest pair.
                                        distance_check_1_mask = dist_xy < distance_xz_block[z + x * block_size];
                                        distance_check_2_mask = dist_xy < distance_yz_block[z + y * block_size];
                                        scalar_xy_closest = distance_check_1_mask & distance_check_2_mask;
                                        xy_reduction += scalar_xy_closest*conflict_yz_block[z + y * block_size];
                                        yx_reduction += scalar_xy_closest*conflict_xz_block[z + x * block_size];

                                        // xz closest pair.
                                        distance_check_1_mask = distance_xz_block[z + x * block_size] < dist_xy;
                                        distance_check_2_mask =  distance_xz_block[z + x * block_size] < distance_yz_block[z + y * block_size];
                                        scalar_xz_closest = distance_check_1_mask & distance_check_2_mask;
                                        cohesion_xz_block[z + x * block_size] += scalar_xz_closest*conflict_yz_block[z + y * block_size];
                                        cohesion_zx_block[z + x * block_size] += scalar_xz_closest*conflict_xy_val;

                                        //yz closest pair.
                                        distance_check_1_mask = distance_yz_block[z + y * block_size] < distance_xz_block[z + x * block_size];
                                        distance_check_2_mask = distance_yz_block[z + y * block_size] < dist_xy;
                                        scalar_yz_closest = distance_check_1_mask & distance_check_2_mask;
                                        cohesion_yz_block[z + y * block_size] += scalar_yz_closest*conflict_xz_block[z + x * block_size];
                                        cohesion_zy_block[z + y * block_size] += scalar_yz_closest*conflict_xy_val;
                                    }
                                    cohesion_xy_block[y + x * block_size] += xy_reduction;
                                    cohesion_yx_block[y + x * block_size] += yx_reduction;
                                }
                            }
                        }
                        // copy CXY, CXZ, CYZ.
                        for(i = 0; i < block_size; ++i){
                            for(j = 0; j < block_size; ++j){
                                idx = j + i * n;
                                C[yb + xb * n + idx] += cohesion_xy_block[j + i * block_size];
                                C[zb + xb * n + idx] += cohesion_xz_block[j + i * block_size];
                                C[zb + yb * n + idx] += cohesion_yz_block[j + i * block_size];
                            }
                        }
                        // copy CYX, CZX, CZY.
                        for(i = 0; i < block_size; ++i){
                            for(j = 0; j < block_size; ++j){
                                idx = j + i * n;
                                C[xb + yb * n + idx] += cohesion_yx_block[i + j * block_size];
                                C[xb + zb * n + idx] += cohesion_zx_block[i + j * block_size];
                                C[yb + zb * n + idx] += cohesion_zy_block[i + j * block_size];
                            }
                        }
                        _mm_free(distance_xy_block); _mm_free(distance_xz_block); _mm_free(distance_yz_block);
                        _mm_free(conflict_xy_block); _mm_free(conflict_xz_block); _mm_free(conflict_yz_block);
                        _mm_free(cohesion_xy_block); _mm_free(cohesion_yx_block);
                        _mm_free(cohesion_xz_block); _mm_free(cohesion_zx_block);
                        _mm_free(cohesion_yz_block); _mm_free(cohesion_zy_block);
                    }
                }
            }
        }
    }
    // print_matrix(n, n, C);
    _mm_free(conflict_matrix);
    cohesion_loop_time = omp_get_wtime() - time_start;
    printf("==============================================\n");
    printf("Triplet OMP Loop Times\n");
    printf("==============================================\n");
    printf("conflict focus size loop time: %.5fs\n", conflict_loop_time);
    printf("cohesion matrix update loop time: %.5fs\n\n", cohesion_loop_time);
}

void pald_triplet_openmp(float *D, float beta, int n, float *C, int conflict_block_size, int cohesion_block_size, int nthreads){
    //TODO: Optimized sequential triplet code.
    int block_size = conflict_block_size;
    float* restrict conflict_matrix = (float *)  _mm_malloc(n * n * sizeof(float), VECALIGN);
    // memset(conflict_matrix, 0, n * n * sizeof(float));
    unsigned int* restrict conflict_matrix_int = (unsigned int*)  _mm_malloc(n * n * sizeof(unsigned int), VECALIGN);
    double time_start = 0.0, conflict_loop_time = 0.0, cohesion_loop_time;
    // memset(conflict_matrix_int, 0, n * n * sizeof(unsigned int));
    time_start = omp_get_wtime();
    #pragma omp parallel shared(D, conflict_matrix_int, conflict_matrix, n) num_threads(nthreads)
    {
        // printf("nthreads: %d\n", omp_get_num_threads());
        unsigned int scalar_xy_closest_int, scalar_xz_closest_int, scalar_yz_closest_int;
        unsigned int distance_check_1_mask, distance_check_2_mask;
        unsigned int xy_reduction_int;

        unsigned int *conflict_xy_int, *conflict_xz_int, *conflict_yz_int;
        float dist_xy  = 0.f;
        float conflict_xy_val = 0.f;
        unsigned int loop_len = 0;

        // char print_out = 0;
        double time_start = 0.0, time_start2 = 0.0;
        unsigned int xb, yb, zb, x, y, z;
        unsigned int i, j, k;
        unsigned int x_block, y_block, z_block;
        // int size_xy = block_size, size_xz = block_size, size_yz = block_size;
        unsigned int xend, ystart, zstart;
        float xy_reduction = 0.f, yx_reduction = 0.f, cohesion_sum = 0.f;
        // compute conflict focus sizes.
        unsigned int iters = 0;
        unsigned int idx;
        #pragma omp for
        for (i = 0; i < n; ++i){
            for (j = i + 1; j < n; ++j){
                // conflict_matrix[j + i * n] = 2.;
                conflict_matrix_int[j + i * n] = 2;
            }
        }
        // print_matrix_int(n, n, conflict_matrix_int);
        // if(print_out)
        //     print_matrix(n, n, conflict_matrix);
        #pragma omp single
        for(xb = 0; xb < n; xb += block_size){
            x_block = ((xb + block_size) < n) ? block_size : (n - xb);
            for(yb = xb; yb < n; yb += block_size){
                y_block = ((yb + block_size) < n) ? block_size : (n - yb);
                for(zb = yb; zb < n; zb += block_size){
                    z_block = ((zb + block_size) < n) ? block_size : (n - zb);
                    // Set DXZ and DYZ blocks.
                    #pragma omp task default(none)\
                    private(i, j, x, y, z, idx, loop_len, dist_xy, xend, ystart, zstart) \
                    private(scalar_xy_closest_int, scalar_xz_closest_int, scalar_yz_closest_int) \
                    private(distance_check_1_mask, distance_check_2_mask, xy_reduction_int) \
                    firstprivate(xb, yb, zb, x_block, y_block, z_block) \
                    shared(block_size, D, n, conflict_matrix_int) \
                    depend(inout: conflict_matrix_int[yb + xb * n]) \
                    depend(inout: conflict_matrix_int[zb + xb * n]) \
                    depend(inout: conflict_matrix_int[zb + yb * n])
                    {
                        float* restrict distance_xy_block = (float*) _mm_malloc(sizeof(float) * block_size * block_size, VECALIGN);
                        float* restrict distance_xz_block = (float*) _mm_malloc(sizeof(float) * block_size * block_size, VECALIGN);
                        float* restrict distance_yz_block = (float*) _mm_malloc(sizeof(float) * block_size * block_size, VECALIGN);

                        unsigned int* restrict conflict_xy_block_int = (unsigned int *) _mm_malloc(sizeof(int) * block_size * block_size, VECALIGN);
                        unsigned int* restrict conflict_xz_block_int = (unsigned int *) _mm_malloc(sizeof(int) * block_size * block_size, VECALIGN);
                        unsigned int* restrict conflict_yz_block_int = (unsigned int *) _mm_malloc(sizeof(int) * block_size * block_size, VECALIGN);

                        // Copy DXY, DXZ, DYZ blocks.
                        for(i = 0; i < x_block; ++i){
                            memcpy(distance_xy_block + i * block_size, D + yb + (xb + i) * n, sizeof(float) * y_block);
                        }
                        for(i = 0; i < x_block; ++i){
                            memcpy(distance_xz_block + i * block_size, D + zb + (xb + i) * n, sizeof(float) * z_block);
                        }

                        for(i = 0; i < y_block; ++i){
                            memcpy(distance_yz_block + i * block_size, D + zb + (yb + i) * n, sizeof(float) * z_block);
                        }
                        // Set UXY, UXZ, and UYZ blocks.
                        memset(conflict_xy_block_int, 0, sizeof(int) * block_size * block_size);
                        memset(conflict_xz_block_int, 0, sizeof(int) * block_size * block_size);
                        memset(conflict_yz_block_int, 0, sizeof(int) * block_size * block_size);
                        xend = (xb == yb && yb == zb) ? x_block - 1 : x_block;
                        for(x = 0; x < xend; ++x){
                            ystart = (xb == yb) ? x + 1 : 0;
                            //if(xb == yb){
                                for(y = ystart; y < y_block; ++y){
                                    // if(yb == zb){
                                    //     zstart = y + 1;
                                    // }
                                    // xy_reduction = 0.f;
                                    xy_reduction_int = 0;
                                    zstart = (yb == zb) ? y + 1 : 0;
                                    dist_xy = distance_xy_block[y + x * block_size];
                                    // contains_tie = 0.f;
                                    loop_len = z_block - zstart;
                                    if(yb == zb){
                                        // for (z = y + 1; z < block_size; ++z){
                                        // #pragma unroll(16)
                                        for (z = 0; z < loop_len; ++z){
                                            idx = z + y + 1;
                                            //compute masks for conflict blocks.
                                            distance_check_1_mask = dist_xy < distance_xz_block[idx + x * block_size];
                                            distance_check_2_mask = dist_xy < distance_yz_block[idx + y * block_size];
                                            scalar_xy_closest_int = distance_check_1_mask & distance_check_2_mask;

                                            distance_check_1_mask = distance_xz_block[idx + x * block_size] < dist_xy;
                                            distance_check_2_mask =  distance_xz_block[idx + x * block_size] < distance_yz_block[idx + y * block_size];
                                            scalar_xz_closest_int = distance_check_1_mask & distance_check_2_mask;

                                            distance_check_1_mask = distance_yz_block[idx + y * block_size] < distance_xz_block[idx + x * block_size];
                                            distance_check_2_mask = distance_yz_block[idx + y * block_size] < dist_xy;
                                            scalar_yz_closest_int = distance_check_1_mask & distance_check_2_mask;


                                            xy_reduction_int += scalar_xz_closest_int + scalar_yz_closest_int;
                                            conflict_yz_block_int[idx + y * block_size] += scalar_xy_closest_int + scalar_xz_closest_int;
                                            conflict_xz_block_int[idx + x * block_size] += scalar_xy_closest_int + scalar_yz_closest_int;
                                        }
                                        conflict_xy_block_int[y + x * block_size] += xy_reduction_int;
                                    }
                                    else{
                                        // __assume_aligned(distance_xz_block, VECALIGN);
                                        // __assume_aligned(distance_yz_block, VECALIGN);
                                        // __assume_aligned(conflict_yz_block_int, VECALIGN);
                                        // __assume_aligned(conflict_xz_block_int, VECALIGN);
                                        // #pragma unroll(16)
                                        for (z = 0; z < z_block; ++z){
                                            //compute masks for conflict blocks.
                                            distance_check_1_mask = dist_xy < distance_xz_block[z + x * block_size];
                                            distance_check_2_mask = dist_xy < distance_yz_block[z + y * block_size];
                                            scalar_xy_closest_int = distance_check_1_mask & distance_check_2_mask;

                                            distance_check_1_mask = distance_xz_block[z + x * block_size] < dist_xy;
                                            distance_check_2_mask =  distance_xz_block[z + x * block_size] < distance_yz_block[z + y * block_size];
                                            scalar_xz_closest_int = distance_check_1_mask & distance_check_2_mask;

                                            distance_check_1_mask = distance_yz_block[z + y * block_size] < distance_xz_block[z + x * block_size];
                                            distance_check_2_mask = distance_yz_block[z + y * block_size] < dist_xy;
                                            scalar_yz_closest_int = distance_check_1_mask & distance_check_2_mask;

                                            xy_reduction_int += scalar_xz_closest_int + scalar_yz_closest_int;
                                            conflict_yz_block_int[z + y * block_size] += scalar_xy_closest_int + scalar_xz_closest_int;
                                            conflict_xz_block_int[z + x * block_size] += scalar_xy_closest_int + scalar_yz_closest_int;
                                        }
                                        conflict_xy_block_int[y + x * block_size] += xy_reduction_int;
                                    }
                                }
                           // }
                            // else{
                            //     for(y = 0; y < y_block; ++y){
                            //         xy_reduction_int = 0;
                            //         zstart = (yb == zb) ? y + 1 : 0;
                            //         dist_xy = distance_xy_block[y + x * block_size];
                            //         // contains_tie = 0.f;
                            //         loop_len = z_block - zstart;
                            //         if(yb == zb){
                            //             #pragma unroll(16)
                            //             for (z = 0; z < loop_len; ++z){
                            //                 idx = z + y + 1;
                            //                 //compute masks for conflict blocks.
                            //                 distance_check_1_mask = dist_xy < distance_xz_block[idx + x * block_size];
                            //                 distance_check_2_mask = dist_xy < distance_yz_block[idx + y * block_size];
                            //                 scalar_xy_closest_int = distance_check_1_mask & distance_check_2_mask;

                            //                 distance_check_1_mask = distance_xz_block[idx + x * block_size] < dist_xy;
                            //                 distance_check_2_mask =  distance_xz_block[idx + x * block_size] < distance_yz_block[idx + y * block_size];
                            //                 scalar_xz_closest_int = distance_check_1_mask & distance_check_2_mask;

                            //                 distance_check_1_mask = distance_yz_block[idx + y * block_size] < distance_xz_block[idx + x * block_size];
                            //                 distance_check_2_mask = distance_yz_block[idx + y * block_size] < dist_xy;
                            //                 scalar_yz_closest_int = distance_check_1_mask & distance_check_2_mask;


                            //                 xy_reduction_int += scalar_xz_closest_int + scalar_yz_closest_int;
                            //                 conflict_yz_block_int[idx + y * block_size] += scalar_xy_closest_int + scalar_xz_closest_int;
                            //                 conflict_xz_block_int[idx + x * block_size] += scalar_xy_closest_int + scalar_yz_closest_int;
                            //             }
                            //             conflict_xy_block_int[y + x * block_size] += xy_reduction_int;
                            //         }
                            //         else{
                            //             #pragma unroll(16)
                            //             for (z = 0; z < z_block; ++z){
                            //                 //compute masks for conflict blocks.
                            //                 distance_check_1_mask = dist_xy < distance_xz_block[z + x * block_size];
                            //                 distance_check_2_mask = dist_xy < distance_yz_block[z + y * block_size];
                            //                 scalar_xy_closest_int = distance_check_1_mask & distance_check_2_mask;

                            //                 distance_check_1_mask = distance_xz_block[z + x * block_size] < dist_xy;
                            //                 distance_check_2_mask =  distance_xz_block[z + x * block_size] < distance_yz_block[z + y * block_size];
                            //                 scalar_xz_closest_int = distance_check_1_mask & distance_check_2_mask;

                            //                 distance_check_1_mask = distance_yz_block[z + y * block_size] < distance_xz_block[z + x * block_size];
                            //                 distance_check_2_mask = distance_yz_block[z + y * block_size] < dist_xy;
                            //                 scalar_yz_closest_int = distance_check_1_mask & distance_check_2_mask;
                            //                 xy_reduction_int += scalar_xz_closest_int + scalar_yz_closest_int;
                            //                 conflict_yz_block_int[z + y * block_size] += scalar_xy_closest_int + scalar_xz_closest_int;
                            //                 conflict_xz_block_int[z + x * block_size] += scalar_xy_closest_int + scalar_yz_closest_int;
                            //             }
                            //             conflict_xy_block_int[y + x * block_size] += xy_reduction_int;
                            //         }
                            //     }
                            // }
                        }
                        // conflict_xy_int = conflict_matrix_int + yb + xb * n;
                        // conflict_xz_int = conflict_matrix_int + zb + xb * n;
                        // conflict_yz_int = conflict_matrix_int + zb + yb * n;
                        for(i = 0; i < x_block; ++i){
                            for(j = 0; j < y_block; ++j){
                                idx = j + i * n;
                                conflict_matrix_int[yb + xb * n + idx] += conflict_xy_block_int[j + i * block_size];
                            }
                        }
                        for(i = 0; i < x_block; ++i){
                            for(j = 0; j < z_block; ++j){
                                idx = j + i * n;
                                conflict_matrix_int[zb + xb * n + idx] += conflict_xz_block_int[j + i * block_size];
                            }
                        }
                        for(i = 0; i < y_block; ++i){
                            for(j = 0; j < z_block; ++j){
                                idx = j + i * n;
                                conflict_matrix_int[zb + yb * n + idx] += conflict_yz_block_int[j + i * block_size];
                            }
                        }
                        // printf("(xb: %d, yb: %d, zb: %d)\n", xb, yb, zb);
                        // print_matrix_int(block_size, block_size, conflict_yz_block_int);
                        // printf("[\n");
                        // for(i = 0; i < block_size; ++i){
                        //     for(j = 0; j < block_size; ++j){
                        //         printf("%d ", conflict_xy_int[j + i * n]);
                        //     }
                        //     printf(";\n");
                        // }
                        // printf("];\n");
                        _mm_free(distance_xy_block); _mm_free(distance_xz_block); _mm_free(distance_yz_block);
                        _mm_free(conflict_xy_block_int); _mm_free(conflict_xz_block_int); _mm_free(conflict_yz_block_int);
                    }
                }
                // print_matrix_int(n,n, conflict_matrix_int);
            }
        }
        // #pragma omp master
        // print_matrix_int(n, n, conflict_matrix_int);
        #pragma omp for
        for(i = 0; i < n * n; ++i){
            // conflict_matrix[i] = 1.f/conflict_matrix[i];
            conflict_matrix[i] = 1.f/conflict_matrix_int[i];
        }
        // #pragma omp master
        // print_matrix_int(n, n, conflict_matrix_int);
        // return;
        // printf("\n\n");
            // initialize diagonal of C.

    }
    // print_matrix(n, n, conflict_matrix);
    _mm_free(conflict_matrix_int);

    conflict_loop_time = omp_get_wtime() - time_start;
    time_start = omp_get_wtime();
    block_size = cohesion_block_size;
    #pragma omp parallel shared(C, D, conflict_matrix, n) num_threads(nthreads)
    {
        unsigned int iters = 0;
        double time_start = 0.0, time_start2 = 0.0;
        unsigned int xb, yb, zb;
        unsigned int x_block, y_block, z_block;
        unsigned int i, j;
        float sum = 0.f;
        unsigned int x, y, z, idx, zstart, ystart, xend, loop_len;
        unsigned int distance_check_1_mask, distance_check_2_mask;
        float dist_xy = 0.f, conflict_xy_val = 0.f;
        float xy_reduction = 0.f, yx_reduction = 0.f, cohesion_sum = 0.f;
        float scalar_xy_closest, scalar_xz_closest, scalar_yz_closest;



        #pragma omp for
        for (i = 0; i < n; ++i){
            sum = 0.f;
            for (j = 0; j < i; ++j){
                sum += conflict_matrix[i + j * n];
            }
            for (j = i + 1; j < n; ++j){
                sum += conflict_matrix[j + i * n];
            }
            C[i + i * n] = sum;
        }
        #pragma omp single
        for(xb = 0; xb < n; xb += block_size){
            x_block = ((xb + block_size) < n) ? block_size : (n - xb);
            for(yb = xb; yb < n; yb += block_size){
                y_block = ((yb + block_size) < n) ? block_size : (n - yb);
                for(zb = yb; zb < n; zb += block_size){
                    z_block = ((zb + block_size) < n) ? block_size : (n - zb);
                    #pragma omp task default(none)\
                    private(i, j, x, y, z, idx, loop_len, dist_xy, xend, ystart, zstart) \
                    private(scalar_xy_closest, scalar_xz_closest, scalar_yz_closest) \
                    private(distance_check_1_mask, distance_check_2_mask, xy_reduction) \
                    private(yx_reduction, conflict_xy_val) \
                    firstprivate(xb, yb, zb, x_block, y_block, z_block) \
                    shared(block_size, D, C, n, conflict_matrix) \
                    depend(inout: C[yb + xb * n]) \
                    depend(inout: C[zb + xb * n]) \
                    depend(inout: C[zb + yb * n]) \
                    depend(inout: C[xb + yb * n]) \
                    depend(inout: C[xb + zb * n]) \
                    depend(inout: C[yb + zb * n])
                    {
                        // if(omp_get_thread_num() == 0){
                        //     printf("tid: %d, (xb: %d, yb:%d, zb:%d)\n", omp_get_thread_num(), xb/block_size, yb/block_size, zb/block_size);
                        // }
                        // ntasks[omp_get_thread_num()]++;
                        // distance matrix cache blocks.
                        float* restrict distance_xy_block = (float*) _mm_malloc(sizeof(float) * block_size * block_size, VECALIGN);
                        float* restrict distance_xz_block = (float*) _mm_malloc(sizeof(float) * block_size * block_size, VECALIGN);
                        float* restrict distance_yz_block = (float*) _mm_malloc(sizeof(float) * block_size * block_size, VECALIGN);
                        // conflict matrix cache blocks.
                        float* restrict conflict_xy_block = (float*) _mm_malloc(sizeof(float) * block_size * block_size, VECALIGN);
                        float* restrict conflict_xz_block = (float*) _mm_malloc(sizeof(float) * block_size * block_size, VECALIGN);
                        float* restrict conflict_yz_block = (float*) _mm_malloc(sizeof(float) * block_size * block_size, VECALIGN);
                        // cohesion matrix cache blocks.
                        float* restrict cohesion_xy_block = (float*) _mm_malloc(sizeof(float) * block_size * block_size, VECALIGN);
                        float* restrict cohesion_yx_block = (float*) _mm_malloc(sizeof(float) * block_size * block_size, VECALIGN);
                        float* restrict cohesion_xz_block = (float*) _mm_malloc(sizeof(float) * block_size * block_size, VECALIGN);
                        float* restrict cohesion_zx_block = (float*) _mm_malloc(sizeof(float) * block_size * block_size, VECALIGN);
                        float* restrict cohesion_yz_block = (float*) _mm_malloc(sizeof(float) * block_size * block_size, VECALIGN);
                        float* restrict cohesion_zy_block = (float*) _mm_malloc(sizeof(float) * block_size * block_size, VECALIGN);
                        // Set distance blocks: DXY, DXZ, DYZ.
                        for(i = 0; i < x_block; ++i){
                            memcpy(distance_xy_block + i * block_size, D + yb + (xb + i) * n, sizeof(float) * y_block);
                            memcpy(conflict_xy_block + i * block_size, conflict_matrix + yb + (xb + i) * n, sizeof(float) * y_block);
                        }
                        // Set conflict blocks: UXY, UXZ, UYZ.
                        for(i = 0; i < x_block; ++i){
                            memcpy(conflict_xz_block + i * block_size, conflict_matrix + zb + (xb + i) * n, sizeof(float) * z_block);
                            memcpy(distance_xz_block + i * block_size, D + zb + (xb + i) * n, sizeof(float) * z_block);
                        }
                        for(i = 0; i < y_block; ++i){
                            memcpy(distance_yz_block + i * block_size, D + zb + (yb + i) * n, sizeof(float) * z_block);
                            memcpy(conflict_yz_block + i * block_size, conflict_matrix + zb + (yb + i) * n, sizeof(float) * z_block);
                        }
                        // Set cohesion blocks: CXY, CYX.
                        memset(cohesion_xy_block, 0, sizeof(float) * block_size * block_size);
                        memset(cohesion_yx_block, 0, sizeof(float) * block_size * block_size);

                        // Set cohesion blocks: CXZ, CZX.
                        memset(cohesion_xz_block, 0, sizeof(float) * block_size * block_size);
                        memset(cohesion_zx_block, 0, sizeof(float) * block_size * block_size);

                        // Set cohesion blocks: CYZ, CZY.
                        memset(cohesion_yz_block, 0, sizeof(float) * block_size * block_size);
                        memset(cohesion_zy_block, 0, sizeof(float) * block_size * block_size);

                        xend = x_block;
                        // ystart = 0;
                        // zstart = 0;
                        if(xb == yb && yb == zb){
                            xend = x_block - 1;
                        }
                        for(x = 0; x < xend; ++x){
                            ystart = (xb == yb) ? x + 1 : 0;
                            for(y = ystart; y < y_block; ++y){
                                xy_reduction = 0.f; yx_reduction = 0.f;
                                dist_xy = distance_xy_block[y + x * block_size];
                                if(yb == zb){
                                    loop_len = z_block - y - 1;
                                    conflict_xy_val = conflict_xy_block[y + x * block_size];
                                    for (z = 0; z < loop_len; ++z){
                                        //compute masks for conflict blocks.
                                        idx = z + y + 1;
                                        distance_check_1_mask = dist_xy < distance_xz_block[idx + x * block_size];
                                        distance_check_2_mask = dist_xy < distance_yz_block[idx + y * block_size];
                                        scalar_xy_closest = distance_check_1_mask & distance_check_2_mask;

                                        distance_check_1_mask = distance_xz_block[idx + x * block_size] < dist_xy;
                                        distance_check_2_mask =  distance_xz_block[idx + x * block_size] < distance_yz_block[idx + y * block_size];
                                        scalar_xz_closest = distance_check_1_mask & distance_check_2_mask;

                                        distance_check_1_mask = distance_yz_block[idx + y * block_size] < distance_xz_block[idx + x * block_size];
                                        distance_check_2_mask = distance_yz_block[idx + y * block_size] < dist_xy;
                                        scalar_yz_closest = distance_check_1_mask & distance_check_2_mask;

                                        // xy closest pair.
                                        yx_reduction += scalar_xy_closest*conflict_xz_block[idx + x * block_size];
                                        xy_reduction += scalar_xy_closest*conflict_yz_block[idx + y * block_size];

                                        // xz closest pair.
                                        cohesion_xz_block[idx + x * block_size] += scalar_xz_closest*conflict_yz_block[idx + y * block_size];
                                        cohesion_zx_block[idx + x * block_size] += scalar_xz_closest*conflict_xy_val;

                                        // yz closest pair.
                                        cohesion_yz_block[idx + y * block_size] += scalar_yz_closest*conflict_xz_block[idx + x * block_size];
                                        cohesion_zy_block[idx + y * block_size] += scalar_yz_closest*conflict_xy_val;
                                    }
                                    cohesion_xy_block[y + x * block_size] += xy_reduction;
                                    cohesion_yx_block[y + x * block_size] += yx_reduction;
                                }
                                else{
                                    conflict_xy_val = conflict_xy_block[y + x * block_size];
                                    //update cohesion blocks.
                                    // #pragma unroll(8)
                                    for (z = 0; z < z_block; ++z){
                                        // xy closest pair.
                                        distance_check_1_mask = dist_xy < distance_xz_block[z + x * block_size];
                                        distance_check_2_mask = dist_xy < distance_yz_block[z + y * block_size];
                                        scalar_xy_closest = distance_check_1_mask & distance_check_2_mask;
                                        xy_reduction += scalar_xy_closest*conflict_yz_block[z + y * block_size];
                                        yx_reduction += scalar_xy_closest*conflict_xz_block[z + x * block_size];

                                        // xz closest pair.
                                        distance_check_1_mask = distance_xz_block[z + x * block_size] < dist_xy;
                                        distance_check_2_mask =  distance_xz_block[z + x * block_size] < distance_yz_block[z + y * block_size];
                                        scalar_xz_closest = distance_check_1_mask & distance_check_2_mask;
                                        cohesion_xz_block[z + x * block_size] += scalar_xz_closest*conflict_yz_block[z + y * block_size];
                                        cohesion_zx_block[z + x * block_size] += scalar_xz_closest*conflict_xy_val;

                                        //yz closest pair.
                                        distance_check_1_mask = distance_yz_block[z + y * block_size] < distance_xz_block[z + x * block_size];
                                        distance_check_2_mask = distance_yz_block[z + y * block_size] < dist_xy;
                                        scalar_yz_closest = distance_check_1_mask & distance_check_2_mask;
                                        cohesion_yz_block[z + y * block_size] += scalar_yz_closest*conflict_xz_block[z + x * block_size];
                                        cohesion_zy_block[z + y * block_size] += scalar_yz_closest*conflict_xy_val;
                                    }
                                    cohesion_xy_block[y + x * block_size] += xy_reduction;
                                    cohesion_yx_block[y + x * block_size] += yx_reduction;
                                }
                            }
                        }
                        // copy CXY, CXZ, CYZ.
                        for(i = 0; i < x_block; ++i){
                            for(j = 0; j < y_block; ++j){
                                idx = j + i * n;
                                C[yb + xb * n + idx] += cohesion_xy_block[j + i * block_size];
                            }
                        }
                        for(i = 0; i < y_block; ++i){
                            for(j = 0; j < x_block; ++j){
                                idx = j + i * n;
                                C[xb + yb * n + idx] += cohesion_yx_block[i + j * block_size];
                            }
                        }
                        for(i = 0; i < x_block; ++i){
                            for(j = 0; j < z_block; ++j){
                                idx = j + i * n;
                                C[zb + xb * n + idx] += cohesion_xz_block[j + i * block_size];
                            }
                        }
                        for(i = 0; i < z_block; ++i){
                            for(j = 0; j < x_block; ++j){
                                idx = j + i * n;
                                C[xb + zb * n + idx] += cohesion_zx_block[i + j * block_size];
                            }
                        }
                        for(i = 0; i < y_block; ++i){
                            for(j = 0; j < z_block; ++j){
                                idx = j + i * n;
                                C[zb + yb * n + idx] += cohesion_yz_block[j + i * block_size];
                            }
                        }
                        for(i = 0; i < z_block; ++i){
                            for(j = 0; j < y_block; ++j){
                                idx = j + i * n;
                                C[yb + zb * n + idx] += cohesion_zy_block[i + j * block_size];
                            }
                        }
                        _mm_free(distance_xy_block); _mm_free(distance_xz_block); _mm_free(distance_yz_block);
                        _mm_free(conflict_xy_block); _mm_free(conflict_xz_block); _mm_free(conflict_yz_block);
                        _mm_free(cohesion_xy_block); _mm_free(cohesion_yx_block);
                        _mm_free(cohesion_xz_block); _mm_free(cohesion_zx_block);
                        _mm_free(cohesion_yz_block); _mm_free(cohesion_zy_block);
                    }
                }
            }
        }
        // printf("tid: %d, ntasks: %d\n", omp_get_thread_num(), ntasks[omp_get_thread_num()]);
    }

    // print_matrix(n, n, C);
    _mm_free(conflict_matrix);
    cohesion_loop_time = omp_get_wtime() - time_start;
    printf("==============================================\n");
    printf("Triplet OMP Loop Times\n");
    printf("==============================================\n");
    printf("conflict focus size loop time: %.5fs\n", conflict_loop_time);
    printf("cohesion matrix update loop time: %.5fs\n\n", cohesion_loop_time);
}

void pald_triplet_intrin_openmp_powersoftwo(float *D, float beta, int n, float *C, int block_size){
    //TODO: Optimized sequential triplet code.
    float* restrict conflict_matrix = (float *)  _mm_malloc(n * n * sizeof(float), VECALIGN);
    // memset(conflict_matrix, 0, n * n * sizeof(float));
    unsigned int* restrict conflict_matrix_int = (unsigned int*)  _mm_malloc(n * n * sizeof(unsigned int), VECALIGN);
    // memset(conflict_matrix_int, 0, n * n * sizeof(unsigned int));
    #pragma omp parallel shared(conflict_matrix_int, conflict_matrix)
    {
        float* restrict distance_xy_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
        float* restrict distance_xz_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
        float* restrict distance_yz_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
        int task;
        // float* restrict mask_xy_closest = (float *) _mm_malloc(block_size * sizeof(float), VECALIGN);
        // float* restrict mask_xz_closest = (float *) _mm_malloc(block_size * sizeof(float), VECALIGN);
        // float* restrict mask_yz_closest = (float *) _mm_malloc(block_size * sizeof(float), VECALIGN);

        // unsigned int* restrict mask_xy_closest_int = (unsigned int*) _mm_malloc(block_size * sizeof(unsigned int), VECALIGN);
        // unsigned int* restrict mask_xz_closest_int = (unsigned int*) _mm_malloc(block_size * sizeof(unsigned int), VECALIGN);
        // unsigned int* restrict mask_yz_closest_int = (unsigned int*) _mm_malloc(block_size * sizeof(unsigned int), VECALIGN);

        unsigned int scalar_xy_closest_int, scalar_xz_closest_int, scalar_yz_closest_int;

        unsigned int* restrict buffer_conflict_xz_block_int = (unsigned int *) _mm_malloc(block_size * block_size * sizeof(unsigned int), VECALIGN);
        unsigned int* restrict buffer_conflict_yz_block_int = (unsigned int *) _mm_malloc(block_size * block_size * sizeof(unsigned int), VECALIGN);
        unsigned int* restrict buffer_conflict_xy_block_int = (unsigned int *) _mm_malloc(block_size * block_size * sizeof(unsigned int), VECALIGN);

        // float* restrict buffer_contains_tie = (float *) _mm_malloc(block_size * sizeof(float), VECALIGN);

        // char distance_check_1 = 0;
        // char distance_check_2 = 0;
        // char distance_check_3 = 0;
        unsigned int distance_check_1_mask, distance_check_2_mask;
        unsigned int xy_reduction_int;
        float dist_xy  = 0.f;
        float conflict_xy_val = 0.f;
        unsigned int loop_len = 0;

        unsigned int *conflict_xy_block_int, *conflict_xz_block_int, *conflict_yz_block_int;
        // char print_out = 0;
        double time_start = 0.0, time_start2 = 0.0;
        double memops_loop_time = 0.0, conflict_loop_time = 0.0, cohesion_loop_time = 0.0;
        int xb, yb, zb, x, y, z;
        int i, j, k;
        // int size_xy = block_size, size_xz = block_size, size_yz = block_size;
        int xend, ystart, zstart;
        float xy_reduction = 0.f, yx_reduction = 0.f, cohesion_sum = 0.f;
        // compute conflict focus sizes.
        int iters = 0;
        int idx;
        time_start = omp_get_wtime();
        #pragma omp for
        for (i = 0; i < n; ++i){
            for (j = i + 1; j < n; ++j){
                // conflict_matrix[j + i * n] = 2.;
                conflict_matrix_int[j + i * n] = 2;
            }
        }
        conflict_loop_time += omp_get_wtime() - time_start;
        // print_matrix_int(n, n, conflict_matrix_int);
        // if(print_out)
        //     print_matrix(n, n, conflict_matrix);

        #pragma omp single nowait
        for(xb = 0; xb < n; xb += block_size){
            printf("nthreads: %d\n", omp_get_thread_num());
            for(yb = xb; yb < n; yb += block_size){
                time_start = omp_get_wtime();

                #pragma omp task \
                shared(n, block_size, D) \
                firstprivate(xb, yb) \
                depend(out: D[yb + xb * n]) \
                depend(in:task)
                {
                    task++;
                    for (i = 0; i < block_size; ++i){
                        memcpy(distance_xy_block + i * block_size, D + yb + (xb + i) * n, sizeof(float)*block_size);
                    }
                    // printf("copying D[xb:%d, yb:%d ]\n", xb, yb);
                    // memset(buffer_conflict_xy_block, 0, sizeof(float)*block_size*block_size);
                    memset(buffer_conflict_xy_block_int, 0, sizeof(int)*block_size*block_size);
                    memops_loop_time += omp_get_wtime() - time_start;
                }
                // copy DXY block from D.
                for(zb = yb; zb < n; zb += block_size){
                    //copy DXZ and DYZ blocks from D.
                    time_start = omp_get_wtime();
                    #pragma omp task \
                    shared(n, block_size, D) \
                    firstprivate(xb, yb, zb) \
                    depend(out: D[zb + xb * n]) \
                    depend(out: D[zb + yb * n]) \
                    depend(inout: task)
                    {
                        task++;
                        for (i = 0; i < block_size; ++i){
                            memcpy(distance_xz_block + i * block_size, D + zb + (xb + i) * n, sizeof(float)*block_size);
                        }
                        for(i = 0; i < block_size; ++i){
                            memcpy(distance_yz_block + i * block_size, D + zb + (yb + i) * n, sizeof(float)*block_size);
                        }

                            // memset(buffer_conflict_xz_block, 0, sizeof(float)*block_size*block_size);
                            // memset(buffer_conflict_yz_block, 0, sizeof(float)*block_size*block_size);
                        memset(buffer_conflict_xz_block_int, 0, sizeof(int)*block_size*block_size);
                        memset(buffer_conflict_yz_block_int, 0, sizeof(int)*block_size*block_size);
                        memops_loop_time += omp_get_wtime() - time_start;
                        // printf("copying D[xb:%d, zb:%d ], D[yb:%d, zb:%d]\n", xb, zb, yb, zb);
                    }
                    // ystart = 0;
                    // zstart = 0;
                    // if(xb == yb && yb == zb){
                    //     xend = block_size - 1;
                    // }
                    #pragma omp task \
                    shared(n, block_size) \
                    firstprivate(xb, yb, zb) \
                    depend(in: D[yb + xb * n]) \
                    depend(in: D[zb + xb * n]) \
                    depend(in: D[zb + yb * n]) \
                    depend(inout: conflict_matrix_int[zb + xb * n]) \
                    depend(inout: conflict_matrix_int[zb + yb * n]) \
                    depend(inout: conflict_matrix_int[yb + xb * n]) \
                    depend(inout: task)
                    {
                        task++;
                        time_start = omp_get_wtime();
                        xend = (xb == yb && yb == zb) ? block_size - 1 : block_size;
                        for(x = 0; x < xend; ++x){
                            // if(xb == yb){
                            //     ystart = x + 1;
                            //     // conflict_yz_block += ystart*n;
                            // }
                            ystart = (xb == yb) ? x + 1 : 0;
                            if(xb == yb){
                                for(y = x + 1; y < block_size; ++y){
                                    // if(yb == zb){
                                    //     zstart = y + 1;
                                    // }
                                    // xy_reduction = 0.f;
                                    xy_reduction_int = 0;
                                    zstart = (yb == zb) ? y + 1 : 0;
                                    dist_xy = distance_xy_block[y + x * block_size];
                                    // contains_tie = 0.f;
                                    loop_len = block_size - zstart;
                                    if(yb == zb){
                                        // for (z = y + 1; z < block_size; ++z){
                                        #pragma unroll(16)
                                        for (z = 0; z < loop_len; ++z){
                                            idx = z + y + 1;
                                            //compute masks for conflict blocks.

                                            distance_check_1_mask = dist_xy < distance_xz_block[idx + x * block_size];
                                            distance_check_2_mask = dist_xy < distance_yz_block[idx + y * block_size];
                                            scalar_xy_closest_int = distance_check_1_mask & distance_check_2_mask;

                                            distance_check_1_mask = distance_xz_block[idx + x * block_size] < dist_xy;
                                            distance_check_2_mask =  distance_xz_block[idx + x * block_size] < distance_yz_block[idx + y * block_size];
                                            scalar_xz_closest_int = distance_check_1_mask & distance_check_2_mask;

                                            distance_check_1_mask = distance_yz_block[idx + y * block_size] < distance_xz_block[idx + x * block_size];
                                            distance_check_2_mask = distance_yz_block[idx + y * block_size] < dist_xy;
                                            scalar_yz_closest_int = distance_check_1_mask & distance_check_2_mask;


                                            xy_reduction_int += scalar_xz_closest_int + scalar_yz_closest_int;
                                            buffer_conflict_yz_block_int[idx + y * block_size] += scalar_xy_closest_int + scalar_xz_closest_int;
                                            buffer_conflict_xz_block_int[idx + x * block_size] += scalar_xy_closest_int + scalar_yz_closest_int;
                                        }

                                        buffer_conflict_xy_block_int[y + x * block_size] += xy_reduction_int;
                                    }
                                    else{
                                        // _mm512_store_epi32(buffer_conflict_xy_block_int + y + x * block_size, conflict_xy);
                                        // __m512 dist_xy_avx = _mm512_set1_ps(dist_xy);
                                        // __m512i all_ones = _mm512_set1_epi32(1);
                                        // __m512 dist_xz_avx, dist_yz_avx;
                                        // __mmask16 cmp_result_1, cmp_result_2, cmp_result_3;
                                        // __m512i conf_xy, conf_xz, conf_yz;
                                        // // for(z = 0; z < block_size; z+=16){
                                        // //     dist_xz_avx = _mm512_load_ps(distance_xz_block + z + x * block_size);
                                        // //     dist_yz_avx = _mm512_load_ps(distance_yz_block + z + y * block_size);
                                        // //     distance_check_1_mask = _mm512_cmplt_ps_mask(dist_xy_avx, dist_xz_avx);
                                        // //     distance_check_1_mask = _mm512_cmplt_ps_mask(dist_xy_avx, dist_yz_avx);
                                        // //     cmp_result_1 = distance_check_1_mask & distance_check_2_mask;

                                        // //     distance_check_1_mask = _mm512_cmplt_ps_mask(dist_xz_avx, dist_xy_avx);
                                        // //     distance_check_1_mask = _mm512_cmplt_ps_mask(dist_xz_avx, dist_yz_avx);
                                        // //     cmp_result_2 = distance_check_1_mask & distance_check_2_mask;

                                        // //     distance_check_1_mask = _mm512_cmplt_ps_mask(dist_yz_avx, dist_xy_avx);
                                        // //     distance_check_1_mask = _mm512_cmplt_ps_mask(dist_yz_avx, dist_xz_avx);
                                        // //     cmp_result_3 = distance_check_1_mask & distance_check_2_mask;

                                        // // }
                                        #pragma unroll(16)
                                        for (z = 0; z < block_size; ++z){
                                            //compute masks for conflict blocks.
                                            distance_check_1_mask = dist_xy < distance_xz_block[z + x * block_size];
                                            distance_check_2_mask = dist_xy < distance_yz_block[z + y * block_size];
                                            scalar_xy_closest_int = distance_check_1_mask & distance_check_2_mask;

                                            distance_check_1_mask = distance_xz_block[z + x * block_size] < dist_xy;
                                            distance_check_2_mask =  distance_xz_block[z + x * block_size] < distance_yz_block[z + y * block_size];
                                            scalar_xz_closest_int = distance_check_1_mask & distance_check_2_mask;

                                            distance_check_1_mask = distance_yz_block[z + y * block_size] < distance_xz_block[z + x * block_size];
                                            distance_check_2_mask = distance_yz_block[z + y * block_size] < dist_xy;
                                            scalar_yz_closest_int = distance_check_1_mask & distance_check_2_mask;

                                            xy_reduction_int += scalar_xz_closest_int + scalar_yz_closest_int;
                                            buffer_conflict_yz_block_int[z + y * block_size] += scalar_xy_closest_int + scalar_xz_closest_int;
                                            buffer_conflict_xz_block_int[z + x * block_size] += scalar_xy_closest_int + scalar_yz_closest_int;
                                        }
                                        buffer_conflict_xy_block_int[y + x * block_size] += xy_reduction_int;
                                    }
                                    // conflict_yz_block += n;
                                }
                            }
                            else{
                                for(y = 0; y < block_size; ++y){
                                    // if(yb == zb){
                                    //     zstart = y + 1;
                                    // }
                                    // xy_reduction = 0.f;
                                    xy_reduction_int = 0;
                                    zstart = (yb == zb) ? y + 1 : 0;
                                    dist_xy = distance_xy_block[y + x * block_size];
                                    // contains_tie = 0.f;
                                    loop_len = block_size - zstart;
                                    if(yb == zb){
                                        #pragma unroll(16)
                                        for (z = 0; z < loop_len; ++z){
                                            idx = z + y + 1;
                                            //compute masks for conflict blocks.

                                            distance_check_1_mask = dist_xy < distance_xz_block[idx + x * block_size];
                                            distance_check_2_mask = dist_xy < distance_yz_block[idx + y * block_size];
                                            scalar_xy_closest_int = distance_check_1_mask & distance_check_2_mask;

                                            distance_check_1_mask = distance_xz_block[idx + x * block_size] < dist_xy;
                                            distance_check_2_mask =  distance_xz_block[idx + x * block_size] < distance_yz_block[idx + y * block_size];
                                            scalar_xz_closest_int = distance_check_1_mask & distance_check_2_mask;

                                            distance_check_1_mask = distance_yz_block[idx + y * block_size] < distance_xz_block[idx + x * block_size];
                                            distance_check_2_mask = distance_yz_block[idx + y * block_size] < dist_xy;
                                            scalar_yz_closest_int = distance_check_1_mask & distance_check_2_mask;


                                            xy_reduction_int += scalar_xz_closest_int + scalar_yz_closest_int;
                                            buffer_conflict_yz_block_int[idx + y * block_size] += scalar_xy_closest_int + scalar_xz_closest_int;
                                            buffer_conflict_xz_block_int[idx + x * block_size] += scalar_xy_closest_int + scalar_yz_closest_int;
                                        }
                                        buffer_conflict_xy_block_int[y + x * block_size] += xy_reduction_int;
                                    }
                                    else{
                                        #pragma unroll(16)
                                        for (z = 0; z < block_size; ++z){
                                            //compute masks for conflict blocks.
                                            distance_check_1_mask = dist_xy < distance_xz_block[z + x * block_size];
                                            distance_check_2_mask = dist_xy < distance_yz_block[z + y * block_size];
                                            scalar_xy_closest_int = distance_check_1_mask & distance_check_2_mask;

                                            distance_check_1_mask = distance_xz_block[z + x * block_size] < dist_xy;
                                            distance_check_2_mask =  distance_xz_block[z + x * block_size] < distance_yz_block[z + y * block_size];
                                            scalar_xz_closest_int = distance_check_1_mask & distance_check_2_mask;

                                            distance_check_1_mask = distance_yz_block[z + y * block_size] < distance_xz_block[z + x * block_size];
                                            distance_check_2_mask = distance_yz_block[z + y * block_size] < dist_xy;
                                            scalar_yz_closest_int = distance_check_1_mask & distance_check_2_mask;
                                            xy_reduction_int += scalar_xz_closest_int + scalar_yz_closest_int;
                                            buffer_conflict_yz_block_int[z + y * block_size] += scalar_xy_closest_int + scalar_xz_closest_int;
                                            buffer_conflict_xz_block_int[z + x * block_size] += scalar_xy_closest_int + scalar_yz_closest_int;
                                        }
                                        buffer_conflict_xy_block_int[y + x * block_size] += xy_reduction_int;
                                    }
                                    // conflict_yz_block += n;
                                }
                            }

                        }
                        conflict_loop_time += omp_get_wtime() - time_start;
                    }
                    // conflict_xy_block = conflict_matrix + yb + xb * n;
                    // conflict_xz_block = conflict_matrix + zb + xb * n;
                    // conflict_yz_block = conflict_matrix + zb + yb * n;
                    #pragma omp task \
                    shared(n, block_size, conflict_matrix_int) \
                    firstprivate(xb, zb) \
                    depend(inout: conflict_matrix_int[zb + xb * n]) \
                    depend(inout: task)
                    {
                        task++;
                        time_start2 = omp_get_wtime();
                        conflict_xz_block_int = conflict_matrix_int + zb + xb * n;
                        for(i = 0; i < block_size; ++i){
                            for(j = 0; j < block_size; ++j){
                                conflict_xz_block_int[j + i * n] += buffer_conflict_xz_block_int[j + i * block_size];
                            }
                        }
                        memops_loop_time += omp_get_wtime() - time_start2;
                    }
                    #pragma omp task \
                    shared(n, block_size, conflict_matrix_int) \
                    firstprivate(yb, zb) \
                    depend(inout: conflict_matrix_int[zb + yb * n]) \
                    depend(inout: task)
                    {
                        task++;
                        time_start2 = omp_get_wtime();
                        conflict_yz_block_int = conflict_matrix_int + zb + yb * n;
                        for(i = 0; i < block_size; ++i){
                            for(j = 0; j < block_size; ++j){
                                conflict_yz_block_int[j + i * n] += buffer_conflict_yz_block_int[j + i * block_size];
                            }

                        }
                        printf("writing Conf[xb:%d, zb:%d ], Conf[yb:%d, zb:%d]\n", xb, zb, yb, zb);
                        print_matrix_int(block_size, n, conflict_xz_block_int);
                        print_matrix_int(block_size, n, conflict_yz_block_int);
                        memops_loop_time += omp_get_wtime() - time_start2;
                    }

                }
                #pragma omp task \
                shared(n, block_size, conflict_matrix_int) \
                firstprivate(xb, yb) \
                depend(inout: conflict_matrix_int[yb + xb * n]) \
                depend(inout: task)
                {
                    task++;
                // conflict_xy_block = conflict_matrix + yb + xb * n;
                    time_start2 = omp_get_wtime();
                    conflict_xy_block_int = conflict_matrix_int + yb + xb * n;
                    for(i = 0; i < block_size; ++i){
                        for(j = 0; j < block_size; ++j){
                            // conflict_xy_block[j + i * n] += buffer_conflict_xy_block[j + i * block_size];
                            conflict_xy_block_int[j + i * n] += buffer_conflict_xy_block_int[j + i * block_size];
                        }
                        // conflict_xy_block += n;
                    }
                    printf("writing Conf[xb:%d, yb:%d ]\n", xb, yb);
                    print_matrix_int(block_size, n, conflict_xy_block_int);
                    memops_loop_time += omp_get_wtime() - time_start2;
                }
            }
        }
        time_start = omp_get_wtime();
        #pragma omp parallel for shared(conflict_matrix, conflict_matrix_int, n)
        for(i = 0; i < n * n; ++i){
            // conflict_matrix[i] = 1.f/conflict_matrix[i];
            conflict_matrix[i] = 1.f/conflict_matrix_int[i];
        }
        print_matrix_int(n, n, conflict_matrix_int);
        conflict_loop_time += omp_get_wtime() - time_start;
        // return;
        // printf("\n\n");
            // initialize diagonal of C.
        float sum;
        time_start = omp_get_wtime();
        #pragma omp parallel for private(i, j, sum) shared(C, conflict_matrix, n)
        for (i = 0; i < n; ++i){
            sum = 0.f;
            for (j = 0; j < i; ++j){
                sum += conflict_matrix[i + j * n];
            }
            for (j = i + 1; j < n; ++j){
                sum += conflict_matrix[j + i * n];
            }
            C[i + i * n] = sum;
        }
        cohesion_loop_time += omp_get_wtime() - time_start;
        iters = 0;
        time_start = omp_get_wtime();
        _mm_free(conflict_matrix_int);
        _mm_free(buffer_conflict_xy_block_int); _mm_free(buffer_conflict_xz_block_int); _mm_free(buffer_conflict_yz_block_int);
        _mm_free(distance_xy_block); _mm_free(distance_xz_block); _mm_free(distance_yz_block);
        memops_loop_time += omp_get_wtime() - time_start;
    }
    return;
        double time_start = 0.0, time_start2 = 0.0;
        time_start = omp_get_wtime();
        double memops_loop_time = 0.0, conflict_loop_time = 0.0, cohesion_loop_time = 0.0;
        int i, j, k, x, y, z, idx, zstart, ystart, xend, loop_len;
        int xb, yb, zb;
        unsigned int distance_check_1_mask, distance_check_2_mask;
        float dist_xy = 0.f, conflict_xy_val = 0.f;
        float xy_reduction = 0.f, yx_reduction = 0.f, cohesion_sum = 0.f;
        float *conflict_xy_block, *conflict_xz_block, *conflict_yz_block;
        float *cohesion_xy_block ;
        float *cohesion_yx_block;
        float *cohesion_xz_block;
        float *cohesion_zx_block;
        float *cohesion_yz_block;
        float *cohesion_zy_block;
        block_size /= 2;

        // float* restrict mask_xy_closest = (float *) _mm_malloc(block_size * sizeof(float), VECALIGN);
        // float* restrict mask_xz_closest = (float *) _mm_malloc(block_size * sizeof(float), VECALIGN);
        // float* restrict mask_yz_closest = (float *) _mm_malloc(block_size * sizeof(float), VECALIGN);

        float scalar_xy_closest, scalar_xz_closest, scalar_yz_closest;

        // float* restrict mask_tie_xy_xz = (float *) _mm_malloc(block_size * sizeof(float), VECALIGN);
        // float* restrict mask_tie_xy_yz = (float *) _mm_malloc(block_size * sizeof(float), VECALIGN);
        // float* restrict mask_tie_xz_yz = (float *) _mm_malloc(block_size * sizeof(float), VECALIGN);

        float* buffer_conflict_xz_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
        float* buffer_conflict_yz_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
        float* buffer_conflict_xy_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);

        float* restrict buffer_zx_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
        float* restrict buffer_zy_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
        float* restrict buffer_xz_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
        float* restrict buffer_yz_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
        float* restrict buffer_xy_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
        float* restrict buffer_yx_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);

        float* distance_xy_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
        float* distance_xz_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
        float* distance_yz_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);

        float* tmp_buffer_conflict_yz_block = buffer_conflict_yz_block;
        float* tmp_distance_yz_block = distance_yz_block;
        float* tmp_buffer_yz_block = buffer_yz_block;
        float* tmp_buffer_zy_block = buffer_zy_block;
        memops_loop_time += omp_get_wtime() - time_start;

        for(xb = 0; xb < n; xb += block_size){
            for(yb = xb; yb < n; yb += block_size){
                time_start = omp_get_wtime();
                for (i = 0; i < block_size; ++i){
                    memcpy(distance_xy_block + i * block_size, D + yb + (xb + i) * n, sizeof(float)*block_size);
                }

                memset(buffer_yx_block,0,sizeof(float)*block_size*block_size);
                memset(buffer_xy_block,0,sizeof(float)*block_size*block_size);
                memops_loop_time += omp_get_wtime() - time_start;
                for(zb = yb; zb < n; zb += block_size){
                    time_start = omp_get_wtime();
                    conflict_xy_block = conflict_matrix + yb + xb * n;
                    conflict_xz_block = conflict_matrix + zb + xb * n;
                    conflict_yz_block = conflict_matrix + zb + yb * n;
                    if(xb == yb){
                        #pragma unroll(8)
                        for (i = 0; i < block_size; ++i){
                            memcpy(distance_xz_block + i * block_size, D + zb + (xb + i) * n, sizeof(float)*block_size);
                            memcpy(buffer_conflict_xz_block + i * block_size, conflict_xz_block + i * n, sizeof(float)*block_size);
                            // memcpy(distance_yz_block + i * block_size, D + zb + (yb + i) * n, sizeof(float)*block_size);
                            // memcpy(buffer_conflict_yz_block + i * block_size, conflict_yz_block + i * n, sizeof(float)*block_size);
                        }
                        distance_yz_block = distance_xz_block;
                        buffer_conflict_yz_block = buffer_conflict_xz_block;
                        memset(buffer_zx_block,0,sizeof(float)*block_size*block_size);
                        memset(buffer_xz_block,0,sizeof(float)*block_size*block_size);

                        buffer_zy_block = buffer_zx_block;
                        buffer_yz_block = buffer_xz_block;
                    }
                    else{
                        distance_yz_block = tmp_distance_yz_block;
                        buffer_conflict_yz_block = tmp_buffer_conflict_yz_block;
                        buffer_zy_block = tmp_buffer_zy_block;
                        buffer_yz_block = tmp_buffer_yz_block;
                        #pragma unroll(8)
                        for (i = 0; i < block_size; ++i){
                            memcpy(distance_xz_block + i * block_size, D + zb + (xb + i) * n, sizeof(float)*block_size);
                            memcpy(buffer_conflict_xz_block + i * block_size, conflict_xz_block + i * n, sizeof(float)*block_size);
                            memcpy(distance_yz_block + i * block_size, D + zb + (yb + i) * n, sizeof(float)*block_size);
                            memcpy(buffer_conflict_yz_block + i * block_size, conflict_yz_block + i * n, sizeof(float)*block_size);
                        }
                        memset(buffer_zx_block,0,sizeof(float)*block_size*block_size);
                        memset(buffer_zy_block,0,sizeof(float)*block_size*block_size);
                        memset(buffer_xz_block,0,sizeof(float)*block_size*block_size);
                        memset(buffer_yz_block,0,sizeof(float)*block_size*block_size);
                    }
                    memops_loop_time += omp_get_wtime() - time_start;

                    time_start = omp_get_wtime();
                    xend = block_size;
                    // ystart = 0;
                    // zstart = 0;
                    if(xb == yb && yb == zb){
                        xend = block_size - 1;
                    }
                    for(x = 0; x < xend; ++x){
                        // if(xb == yb){
                        //     ystart = x + 1;

                        // }
                        ystart = (xb == yb) ? x + 1 : 0;
                        for(y = ystart; y < block_size; ++y){
                            xy_reduction = 0.f; yx_reduction = 0.f;
                            // if(yb == zb){
                            //     zstart = y + 1;
                            // }
                                // zstart = (yb == zb) ? y + 1 : 0;
                            dist_xy = distance_xy_block[y + x * block_size];
                            // loop_len = block_size - zstart;
                            if(yb == zb){
                                loop_len = block_size - y - 1;
                                conflict_xy_val = conflict_xy_block[y];
                                // for (z = y + 1; z < block_size; ++z){
                                for (z = 0; z < loop_len; ++z){
                                    //compute masks for conflict blocks.
                                    idx = z + y + 1;
                                    distance_check_1_mask = dist_xy < distance_xz_block[idx + x * block_size];
                                    distance_check_2_mask = dist_xy < distance_yz_block[idx + y * block_size];
                                    scalar_xy_closest = distance_check_1_mask & distance_check_2_mask;

                                    distance_check_1_mask = distance_xz_block[idx + x * block_size] < dist_xy;
                                    distance_check_2_mask =  distance_xz_block[idx + x * block_size] < distance_yz_block[idx + y * block_size];
                                    scalar_xz_closest = distance_check_1_mask & distance_check_2_mask;

                                    distance_check_1_mask = distance_yz_block[idx + y * block_size] < distance_xz_block[idx + x * block_size];
                                    distance_check_2_mask = distance_yz_block[idx + y * block_size] < dist_xy;
                                    scalar_yz_closest = distance_check_1_mask & distance_check_2_mask;

                                    // xy closest pair.
                                    yx_reduction += scalar_xy_closest*buffer_conflict_xz_block[idx + x * block_size];
                                    xy_reduction += scalar_xy_closest*buffer_conflict_yz_block[idx + y * block_size];

                                    // xz closest pair.
                                    buffer_xz_block[idx + x * block_size] += scalar_xz_closest*buffer_conflict_yz_block[idx + y * block_size];
                                    buffer_zx_block[idx + x * block_size] += scalar_xz_closest*conflict_xy_val;

                                    // yz closest pair.
                                    buffer_yz_block[idx + y * block_size] += scalar_yz_closest*buffer_conflict_xz_block[idx + x * block_size];
                                    buffer_zy_block[idx + y * block_size] += scalar_yz_closest*conflict_xy_val;
                                }
                                buffer_xy_block[y + x * block_size] += xy_reduction;
                                buffer_yx_block[y + x * block_size] += yx_reduction;
                            }
                            else{
                                conflict_xy_val = conflict_xy_block[y];
                                // __m512 all_ones = _mm512_set1_ps(1.f);
                                // __m512 dist_xy_avx = _mm512_set1_ps(dist_xy);
                                // __m512 conflict_xy_avx = _mm512_set1_ps(conflict_xy_val);

                                // __m512 xy_reduction_avx = _mm512_setzero();
                                // __m512 yx_reduction_avx = _mm512_setzero();
                                // __m512 dist_xz_avx, dist_yz_avx, conflict_xz_avx, conflict_yz_avx;
                                // __m512 cohesion_xz_avx, cohesion_zx_avx, cohesion_yz_avx, cohesion_zy_avx;
                                // __mmask16 mask_xy_closest, mask_xz_closest, mask_yz_closest;

                                // __m512 xy_reduction_avx_1 = _mm512_setzero();
                                // __m512 yx_reduction_avx_1 = _mm512_setzero();
                                // __m512 dist_xz_avx_1, dist_yz_avx_1, conflict_xz_avx_1, conflict_yz_avx_1;
                                // __m512 cohesion_xz_avx_1, cohesion_zx_avx_1, cohesion_yz_avx_1, cohesion_zy_avx_1;
                                // __mmask16 mask_xy_closest_1, mask_xz_closest_1, mask_yz_closest_1;

                                // __m512 xy_reduction_avx_2 = _mm512_setzero();
                                // __m512 yx_reduction_avx_2 = _mm512_setzero();
                                // __m512 dist_xz_avx_2, dist_yz_avx_2, conflict_xz_avx_2, conflict_yz_avx_2;
                                // __m512 cohesion_xz_avx_2, cohesion_zx_avx_2, cohesion_yz_avx_2, cohesion_zy_avx_2;
                                // __mmask16 mask_xy_closest_2, mask_xz_closest_2, mask_yz_closest_2;

                                // __m512 xy_reduction_avx_3 = _mm512_setzero();
                                // __m512 yx_reduction_avx_3 = _mm512_setzero();
                                // __m512 dist_xz_avx_3, dist_yz_avx_3, conflict_xz_avx_3, conflict_yz_avx_3;
                                // __m512 cohesion_xz_avx_3, cohesion_zx_avx_3, cohesion_yz_avx_3, cohesion_zy_avx_3;
                                // __mmask16 mask_xy_closest_3, mask_xz_closest_3, mask_yz_closest_3;

                                // __mmask16 distance_check_3_mask, distance_check_4_mask;
                                // __mmask16 distance_check_5_mask, distance_check_6_mask;
                                // __mmask16 distance_check_7_mask, distance_check_8_mask;

                                // for(z = 0; z < block_size; z += 128){
                                //     dist_xz_avx = _mm512_load_ps(distance_xz_block + z + x * block_size);
                                //     dist_yz_avx = _mm512_load_ps(distance_yz_block + z + y * block_size);
                                //     cohesion_xz_avx = _mm512_load_ps(buffer_xz_block + z + x * block_size);
                                //     cohesion_zx_avx = _mm512_load_ps(buffer_zx_block + z + x * block_size);
                                //     cohesion_yz_avx = _mm512_load_ps(buffer_yz_block + z + y * block_size);
                                //     cohesion_zy_avx = _mm512_load_ps(buffer_zy_block + z + y * block_size);
                                //     conflict_yz_avx = _mm512_load_ps(buffer_conflict_yz_block + z + y * block_size);
                                //     conflict_xz_avx = _mm512_load_ps(buffer_conflict_xz_block + z + x * block_size);

                                //     dist_xz_avx_1 = _mm512_load_ps(distance_xz_block + z + 16 + x * block_size);
                                //     dist_yz_avx_1 = _mm512_load_ps(distance_yz_block + z + 16 + y * block_size);
                                //     cohesion_xz_avx_1 = _mm512_load_ps(buffer_xz_block + z + 16 + x * block_size);
                                //     cohesion_zx_avx_1 = _mm512_load_ps(buffer_zx_block + z + 16 + x * block_size);
                                //     cohesion_yz_avx_1 = _mm512_load_ps(buffer_yz_block + z + 16 + y * block_size);
                                //     cohesion_zy_avx_1 = _mm512_load_ps(buffer_zy_block + z + 16 + y * block_size);
                                //     conflict_yz_avx_1 = _mm512_load_ps(buffer_conflict_yz_block + z + 16 + y * block_size);
                                //     conflict_xz_avx_1 = _mm512_load_ps(buffer_conflict_xz_block + z + 16 + x * block_size);

                                //     dist_xz_avx_2 = _mm512_load_ps(distance_xz_block + z + 32 + x * block_size);
                                //     dist_yz_avx_2 = _mm512_load_ps(distance_yz_block + z + 32 + y * block_size);
                                //     cohesion_xz_avx_2 = _mm512_load_ps(buffer_xz_block + z + 32 + x * block_size);
                                //     cohesion_zx_avx_2 = _mm512_load_ps(buffer_zx_block + z + 32 + x * block_size);
                                //     cohesion_yz_avx_2 = _mm512_load_ps(buffer_yz_block + z + 32 + y * block_size);
                                //     cohesion_zy_avx_2 = _mm512_load_ps(buffer_zy_block + z + 32 + y * block_size);
                                //     conflict_yz_avx_2 = _mm512_load_ps(buffer_conflict_yz_block + z + 32 + y * block_size);
                                //     conflict_xz_avx_2 = _mm512_load_ps(buffer_conflict_xz_block + z + 32 + x * block_size);

                                //     dist_xz_avx_3 = _mm512_load_ps(distance_xz_block + z + 64 + x * block_size);
                                //     dist_yz_avx_3 = _mm512_load_ps(distance_yz_block + z + 64 + y * block_size);
                                //     cohesion_xz_avx_3 = _mm512_load_ps(buffer_xz_block + z + 64 + x * block_size);
                                //     cohesion_zx_avx_3 = _mm512_load_ps(buffer_zx_block + z + 64 + x * block_size);
                                //     cohesion_yz_avx_3 = _mm512_load_ps(buffer_yz_block + z + 64 + y * block_size);
                                //     cohesion_zy_avx_3 = _mm512_load_ps(buffer_zy_block + z + 64 + y * block_size);
                                //     conflict_yz_avx_3 = _mm512_load_ps(buffer_conflict_yz_block + z + 64 + y * block_size);
                                //     conflict_xz_avx_3 = _mm512_load_ps(buffer_conflict_xz_block + z + 64 + x * block_size);

                                //     distance_check_1_mask = _mm512_cmplt_ps_mask(dist_xy_avx, dist_xz_avx);
                                //     distance_check_2_mask = _mm512_cmplt_ps_mask(dist_xy_avx, dist_yz_avx);
                                //     mask_xy_closest = distance_check_1_mask & distance_check_2_mask;

                                //     distance_check_1_mask = _mm512_cmplt_ps_mask(dist_xz_avx, dist_xy_avx);
                                //     distance_check_2_mask = _mm512_cmplt_ps_mask(dist_xz_avx, dist_yz_avx);
                                //     mask_xz_closest = distance_check_1_mask & distance_check_2_mask;

                                //     distance_check_1_mask = _mm512_cmplt_ps_mask(dist_yz_avx, dist_xz_avx);
                                //     distance_check_2_mask = _mm512_cmplt_ps_mask(dist_yz_avx, dist_xy_avx);
                                //     mask_yz_closest = distance_check_1_mask & distance_check_2_mask;

                                //     distance_check_3_mask = _mm512_cmplt_ps_mask(dist_xy_avx, dist_xz_avx);
                                //     distance_check_4_mask = _mm512_cmplt_ps_mask(dist_xy_avx, dist_yz_avx);
                                //     mask_xy_closest_1 = distance_check_3_mask & distance_check_4_mask;

                                //     distance_check_3_mask = _mm512_cmplt_ps_mask(dist_xz_avx, dist_xy_avx);
                                //     distance_check_4_mask = _mm512_cmplt_ps_mask(dist_xz_avx, dist_yz_avx);
                                //     mask_xz_closest_1 = distance_check_3_mask & distance_check_4_mask;

                                //     distance_check_3_mask = _mm512_cmplt_ps_mask(dist_yz_avx, dist_xz_avx);
                                //     distance_check_4_mask = _mm512_cmplt_ps_mask(dist_yz_avx, dist_xy_avx);
                                //     mask_yz_closest_1 = distance_check_3_mask & distance_check_4_mask;

                                //     distance_check_5_mask = _mm512_cmplt_ps_mask(dist_xy_avx, dist_xz_avx);
                                //     distance_check_6_mask = _mm512_cmplt_ps_mask(dist_xy_avx, dist_yz_avx);
                                //     mask_xy_closest_2 = distance_check_5_mask & distance_check_6_mask;

                                //     distance_check_5_mask = _mm512_cmplt_ps_mask(dist_xz_avx, dist_xy_avx);
                                //     distance_check_6_mask = _mm512_cmplt_ps_mask(dist_xz_avx, dist_yz_avx);
                                //     mask_xz_closest_2 = distance_check_5_mask & distance_check_6_mask;

                                //     distance_check_5_mask = _mm512_cmplt_ps_mask(dist_yz_avx, dist_xz_avx);
                                //     distance_check_6_mask = _mm512_cmplt_ps_mask(dist_yz_avx, dist_xy_avx);
                                //     mask_yz_closest_2 = distance_check_5_mask & distance_check_6_mask;

                                //     distance_check_7_mask = _mm512_cmplt_ps_mask(dist_xy_avx, dist_xz_avx);
                                //     distance_check_8_mask = _mm512_cmplt_ps_mask(dist_xy_avx, dist_yz_avx);
                                //     mask_xy_closest_3 = distance_check_7_mask & distance_check_8_mask;

                                //     distance_check_7_mask = _mm512_cmplt_ps_mask(dist_xz_avx, dist_xy_avx);
                                //     distance_check_8_mask = _mm512_cmplt_ps_mask(dist_xz_avx, dist_yz_avx);
                                //     mask_xz_closest_3 = distance_check_7_mask & distance_check_8_mask;

                                //     distance_check_7_mask = _mm512_cmplt_ps_mask(dist_yz_avx, dist_xz_avx);
                                //     distance_check_8_mask = _mm512_cmplt_ps_mask(dist_yz_avx, dist_xy_avx);
                                //     mask_yz_closest_3 = distance_check_7_mask & distance_check_8_mask;


                                //     xy_reduction_avx = _mm512_mask_fmadd_ps(all_ones, mask_xy_closest, conflict_yz_avx, xy_reduction_avx);
                                //     yx_reduction_avx = _mm512_mask_fmadd_ps(all_ones, mask_xy_closest, conflict_xz_avx, yx_reduction_avx);

                                //     xy_reduction_avx_1 = _mm512_mask_fmadd_ps(all_ones, mask_xy_closest_1, conflict_yz_avx_1, xy_reduction_avx_1);
                                //     yx_reduction_avx_1 = _mm512_mask_fmadd_ps(all_ones, mask_xy_closest_1, conflict_xz_avx_1, yx_reduction_avx_1);

                                //     xy_reduction_avx_2 = _mm512_mask_fmadd_ps(all_ones, mask_xy_closest_2, conflict_yz_avx_2, xy_reduction_avx_2);
                                //     yx_reduction_avx_2 = _mm512_mask_fmadd_ps(all_ones, mask_xy_closest_2, conflict_xz_avx_2, yx_reduction_avx_2);

                                //     xy_reduction_avx_3 = _mm512_mask_fmadd_ps(all_ones, mask_xy_closest_3, conflict_yz_avx_3, xy_reduction_avx_3);
                                //     yx_reduction_avx_3 = _mm512_mask_fmadd_ps(all_ones, mask_xy_closest_3, conflict_xz_avx_3, yx_reduction_avx_3);

                                //     _mm512_store_ps(buffer_xz_block + z + x * block_size, _mm512_mask_fmadd_ps(all_ones, mask_xz_closest, conflict_yz_avx, cohesion_xz_avx));
                                //     _mm512_store_ps(buffer_zx_block + z + x * block_size, _mm512_mask_fmadd_ps(all_ones, mask_xz_closest, conflict_xy_avx, cohesion_zx_avx));
                                //     _mm512_store_ps(buffer_yz_block + z + y * block_size, _mm512_mask_fmadd_ps(all_ones, mask_yz_closest, conflict_xz_avx, cohesion_yz_avx));
                                //     _mm512_store_ps(buffer_zy_block + z + y * block_size, _mm512_mask_fmadd_ps(all_ones, mask_yz_closest, conflict_xy_avx, cohesion_zy_avx));

                                //     _mm512_store_ps(buffer_xz_block + z + 16 + x * block_size, _mm512_mask_fmadd_ps(all_ones, mask_xz_closest_1, conflict_yz_avx_1, cohesion_xz_avx_1));
                                //     _mm512_store_ps(buffer_zx_block + z + 16 + x * block_size, _mm512_mask_fmadd_ps(all_ones, mask_xz_closest_1, conflict_xy_avx, cohesion_zx_avx_1));
                                //     _mm512_store_ps(buffer_yz_block + z + 16 + y * block_size, _mm512_mask_fmadd_ps(all_ones, mask_yz_closest_1, conflict_xz_avx_1, cohesion_yz_avx_1));
                                //     _mm512_store_ps(buffer_zy_block + z + 16 + y * block_size, _mm512_mask_fmadd_ps(all_ones, mask_yz_closest_1, conflict_xy_avx, cohesion_zy_avx_1));

                                //     _mm512_store_ps(buffer_xz_block + z + 32 + x * block_size, _mm512_mask_fmadd_ps(all_ones, mask_xz_closest_2, conflict_yz_avx_2, cohesion_xz_avx_2));
                                //     _mm512_store_ps(buffer_zx_block + z + 32 + x * block_size, _mm512_mask_fmadd_ps(all_ones, mask_xz_closest_2, conflict_xy_avx, cohesion_zx_avx_2));
                                //     _mm512_store_ps(buffer_yz_block + z + 32 + y * block_size, _mm512_mask_fmadd_ps(all_ones, mask_yz_closest_2, conflict_xz_avx_2, cohesion_yz_avx_2));
                                //     _mm512_store_ps(buffer_zy_block + z + 32 + y * block_size, _mm512_mask_fmadd_ps(all_ones, mask_yz_closest_2, conflict_xy_avx, cohesion_zy_avx_2));

                                //     _mm512_store_ps(buffer_xz_block + z + 64 + x * block_size, _mm512_mask_fmadd_ps(all_ones, mask_xz_closest_3, conflict_yz_avx_3, cohesion_xz_avx_3));
                                //     _mm512_store_ps(buffer_zx_block + z + 64 + x * block_size, _mm512_mask_fmadd_ps(all_ones, mask_xz_closest_3, conflict_xy_avx, cohesion_zx_avx_3));
                                //     _mm512_store_ps(buffer_yz_block + z + 64 + y * block_size, _mm512_mask_fmadd_ps(all_ones, mask_yz_closest_3, conflict_xz_avx_3, cohesion_yz_avx_3));
                                //     _mm512_store_ps(buffer_zy_block + z + 64 + y * block_size, _mm512_mask_fmadd_ps(all_ones, mask_yz_closest_3, conflict_xy_avx, cohesion_zy_avx_3));
                                // }
                                // xy_reduction_avx += xy_reduction_avx_1 + xy_reduction_avx_2 + xy_reduction_avx_3;
                                // yx_reduction_avx += yx_reduction_avx_1 + yx_reduction_avx_2 + yx_reduction_avx_3;
                                // xy_reduction += _mm512_reduce_add_ps(xy_reduction_avx);
                                // yx_reduction += _mm512_reduce_add_ps(yx_reduction_avx);
                                #pragma unroll(8)
                                //update cohesion blocks.
                                for (z = 0; z < block_size; ++z){
                                    // xy closest pair.
                                    distance_check_1_mask = dist_xy < distance_xz_block[z + x * block_size];
                                    distance_check_2_mask = dist_xy < distance_yz_block[z + y * block_size];
                                    scalar_xy_closest = distance_check_1_mask & distance_check_2_mask;
                                    xy_reduction += scalar_xy_closest*buffer_conflict_yz_block[z + y * block_size];
                                    yx_reduction += scalar_xy_closest*buffer_conflict_xz_block[z + x * block_size];

                                    // xz closest pair.
                                    distance_check_1_mask = distance_xz_block[z + x * block_size] < dist_xy;
                                    distance_check_2_mask =  distance_xz_block[z + x * block_size] < distance_yz_block[z + y * block_size];
                                    scalar_xz_closest = distance_check_1_mask & distance_check_2_mask;
                                    buffer_xz_block[z + x * block_size] += scalar_xz_closest*buffer_conflict_yz_block[z + y * block_size];
                                    buffer_zx_block[z + x * block_size] += scalar_xz_closest*conflict_xy_val;

                                    //yz closest pair.
                                    distance_check_1_mask = distance_yz_block[z + y * block_size] < distance_xz_block[z + x * block_size];
                                    distance_check_2_mask = distance_yz_block[z + y * block_size] < dist_xy;
                                    scalar_yz_closest = distance_check_1_mask & distance_check_2_mask;
                                    buffer_yz_block[z + y * block_size] += scalar_yz_closest*buffer_conflict_xz_block[z + x * block_size];
                                    buffer_zy_block[z + y * block_size] += scalar_yz_closest*conflict_xy_val;
                                }
                                buffer_xy_block[y + x * block_size] += xy_reduction;
                                buffer_yx_block[y + x * block_size] += yx_reduction;
                            }
                        }
                        conflict_xy_block += n;
                    }


                    time_start2 = omp_get_wtime();
                    if(xb == yb){
                        cohesion_zx_block = C + xb + zb * n;
                        cohesion_xz_block = C + zb + xb * n;
                        for(i = 0; i < block_size; ++i){
                            for(j = 0; j < block_size; ++j){
                                // printf("idx: %d\n", n*j + i);
                                cohesion_zx_block[j] += buffer_zx_block[i + j * block_size];
                                // cohesion_yx_block[j] += buffer_yx_block[i + j * block_size];
                            }
                            cohesion_zx_block += n;
                        }
                        for(i = 0; i < block_size; ++i){
                            for(j = 0; j < block_size; ++j){
                                cohesion_xz_block[j] += buffer_xz_block[j + i * block_size];
                                // cohesion_xy_block[j] += buffer_xy_block[j + i * block_size];
                            }
                            cohesion_xz_block += n;
                        }

                    }
                    else{
                        cohesion_zx_block = C + xb + zb * n;
                        cohesion_zy_block = C + yb + zb * n;
                        cohesion_xz_block = C + zb + xb * n;
                        cohesion_yz_block = C + zb + yb * n;
                        for(i = 0; i < block_size; ++i){
                            for(j = 0; j < block_size; ++j){
                                // printf("idx: %d\n", n*j + i);
                                cohesion_zx_block[j] += buffer_zx_block[i + j * block_size];
                                cohesion_zy_block[j] += buffer_zy_block[i + j * block_size];
                                // cohesion_yx_block[j] += buffer_yx_block[i + j * block_size];
                            }
                            cohesion_zx_block += n;
                            cohesion_zy_block += n;
                        }
                        for(i = 0; i < block_size; ++i){
                            for(j = 0; j < block_size; ++j){
                                cohesion_xz_block[j] += buffer_xz_block[j + i * block_size];
                                cohesion_yz_block[j] += buffer_yz_block[j + i * block_size];
                                // cohesion_xy_block[j] += buffer_xy_block[j + i * block_size];
                            }
                            cohesion_xz_block += n;
                            cohesion_yz_block += n;
                        }
                    }

                    memops_loop_time += omp_get_wtime() - time_start2;
                    cohesion_loop_time += omp_get_wtime() - time_start;
                }
                time_start2 = omp_get_wtime();
                cohesion_xy_block = C + yb + xb * n;
                cohesion_yx_block = C + xb + yb * n;
                for(i = 0; i < block_size; ++i){
                    for(j = 0; j < block_size; ++j){
                        // printf("idx: %d\n", n*j + i);
                        cohesion_yx_block[j] += buffer_yx_block[i + j * block_size];
                    }
                    cohesion_yx_block += n;
                }
                cohesion_yx_block = C + xb + yb * n;

                conflict_xy_block = conflict_matrix + yb + xb * n;

                for(i = 0; i < block_size; ++i){
                    for(j = 0; j < block_size; ++j){
                        cohesion_xy_block[j] += buffer_xy_block[j + i * block_size];
                    }
                    cohesion_xy_block += n;
                }
                memops_loop_time += omp_get_wtime() - time_start2;
            }
        }
        // print_matrix(n, n, C);

        printf("==============================================\n");
        printf("Seq. Triplet Intrinsics OMP Loop Times\n");
        printf("==============================================\n");

        printf("memops loop time: %.5fs\n", memops_loop_time);
        printf("conflict focus size loop time: %.5fs\n", conflict_loop_time);
        printf("cohesion matrix update loop time: %.5fs\n\n", cohesion_loop_time);

        _mm_free(distance_xy_block); _mm_free(distance_xz_block); _mm_free(tmp_distance_yz_block);
        // _mm_free(mask_tie_xy_xz); _mm_free(mask_tie_xy_yz); _mm_free(mask_tie_xz_yz);
        _mm_free(buffer_zx_block); _mm_free(tmp_buffer_zy_block); _mm_free(buffer_yx_block);
        _mm_free(buffer_xz_block); _mm_free(tmp_buffer_yz_block); _mm_free(buffer_xy_block);
        // _mm_free(mask_xy_closest); _mm_free(mask_xz_closest); _mm_free(mask_yz_closest);
        _mm_free(buffer_conflict_xz_block); _mm_free(tmp_buffer_conflict_yz_block); _mm_free(buffer_conflict_xy_block);
        _mm_free(conflict_matrix);
        // _mm_free(conflict_matrix_int);
        // _mm_free(mask_xy_closest_int); _mm_free(mask_xz_closest_int); _mm_free(mask_yz_closest_int);
        // _mm_free(buffer_conflict_xz_block_int); _mm_free(buffer_conflict_yz_block_int); _mm_free(buffer_conflict_xy_block_int);

}

void pald_triplet_intrin_openmp(float *D, float beta, int n, float *C, int block_size){
    //TODO: Optimized sequential triplet code.
    float* restrict conflict_matrix = (float *)  _mm_malloc(n * n * sizeof(float), VECALIGN);
    // memset(conflict_matrix, 0, n * n * sizeof(float));
    unsigned int* restrict conflict_matrix_int = (unsigned int*)  _mm_malloc(n * n * sizeof(unsigned int), VECALIGN);
    // memset(conflict_matrix_int, 0, n * n * sizeof(unsigned int));
    #pragma omp parallel shared(conflict_matrix_int, conflict_matrix)
    {
        float* restrict distance_xy_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
        float* restrict distance_xz_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
        float* restrict distance_yz_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
        int task;
        // float* restrict mask_xy_closest = (float *) _mm_malloc(block_size * sizeof(float), VECALIGN);
        // float* restrict mask_xz_closest = (float *) _mm_malloc(block_size * sizeof(float), VECALIGN);
        // float* restrict mask_yz_closest = (float *) _mm_malloc(block_size * sizeof(float), VECALIGN);

        // unsigned int* restrict mask_xy_closest_int = (unsigned int*) _mm_malloc(block_size * sizeof(unsigned int), VECALIGN);
        // unsigned int* restrict mask_xz_closest_int = (unsigned int*) _mm_malloc(block_size * sizeof(unsigned int), VECALIGN);
        // unsigned int* restrict mask_yz_closest_int = (unsigned int*) _mm_malloc(block_size * sizeof(unsigned int), VECALIGN);

        unsigned int scalar_xy_closest_int, scalar_xz_closest_int, scalar_yz_closest_int;

        unsigned int* restrict buffer_conflict_xz_block_int = (unsigned int *) _mm_malloc(block_size * block_size * sizeof(unsigned int), VECALIGN);
        unsigned int* restrict buffer_conflict_yz_block_int = (unsigned int *) _mm_malloc(block_size * block_size * sizeof(unsigned int), VECALIGN);
        unsigned int* restrict buffer_conflict_xy_block_int = (unsigned int *) _mm_malloc(block_size * block_size * sizeof(unsigned int), VECALIGN);

        // float* restrict buffer_contains_tie = (float *) _mm_malloc(block_size * sizeof(float), VECALIGN);

        // char distance_check_1 = 0;
        // char distance_check_2 = 0;
        // char distance_check_3 = 0;
        unsigned int distance_check_1_mask, distance_check_2_mask;
        unsigned int xy_reduction_int;
        float dist_xy  = 0.f;
        float conflict_xy_val = 0.f;
        unsigned int loop_len = 0;

        unsigned int *conflict_xy_block_int, *conflict_xz_block_int, *conflict_yz_block_int;
        // char print_out = 0;
        double time_start = 0.0, time_start2 = 0.0;
        double memops_loop_time = 0.0, conflict_loop_time = 0.0, cohesion_loop_time = 0.0;
        int xb, yb, zb, x, y, z;
        int i, j, k;
        int x_block, y_block, z_block;
        // int size_xy = block_size, size_xz = block_size, size_yz = block_size;
        int xend, ystart, zstart;
        float xy_reduction = 0.f, yx_reduction = 0.f, cohesion_sum = 0.f;
        // compute conflict focus sizes.
        int iters = 0;
        int idx;
        time_start = omp_get_wtime();
        #pragma omp for
        for (i = 0; i < n; ++i){
            for (j = i + 1; j < n; ++j){
                // conflict_matrix[j + i * n] = 2.;
                conflict_matrix_int[j + i * n] = 2;
            }
        }
        conflict_loop_time += omp_get_wtime() - time_start;
        // print_matrix_int(n, n, conflict_matrix_int);
        // if(print_out)
        //     print_matrix(n, n, conflict_matrix);

        #pragma omp single nowait
        for(xb = 0; xb < n; xb += block_size){
            printf("nthreads: %d\n", omp_get_thread_num());
            for(yb = xb; yb < n; yb += block_size){
                time_start = omp_get_wtime();

                #pragma omp task \
                shared(n, block_size, D) \
                firstprivate(xb, yb) \
                depend(out: D[yb + xb * n]) \
                depend(in:task)
                {
                    task++;
                    for (i = 0; i < block_size; ++i){
                        memcpy(distance_xy_block + i * block_size, D + yb + (xb + i) * n, sizeof(float)*block_size);
                    }
                    // printf("copying D[xb:%d, yb:%d ]\n", xb, yb);
                    // memset(buffer_conflict_xy_block, 0, sizeof(float)*block_size*block_size);
                    memset(buffer_conflict_xy_block_int, 0, sizeof(int)*block_size*block_size);
                    memops_loop_time += omp_get_wtime() - time_start;
                }
                // copy DXY block from D.
                for(zb = yb; zb < n; zb += block_size){
                    //copy DXZ and DYZ blocks from D.
                    time_start = omp_get_wtime();
                    #pragma omp task \
                    shared(n, block_size, D) \
                    firstprivate(xb, yb, zb) \
                    depend(out: D[zb + xb * n]) \
                    depend(out: D[zb + yb * n]) \
                    depend(inout: task)
                    {
                        task++;
                        for (i = 0; i < block_size; ++i){
                            memcpy(distance_xz_block + i * block_size, D + zb + (xb + i) * n, sizeof(float)*block_size);
                        }
                        for(i = 0; i < block_size; ++i){
                            memcpy(distance_yz_block + i * block_size, D + zb + (yb + i) * n, sizeof(float)*block_size);
                        }

                            // memset(buffer_conflict_xz_block, 0, sizeof(float)*block_size*block_size);
                            // memset(buffer_conflict_yz_block, 0, sizeof(float)*block_size*block_size);
                        memset(buffer_conflict_xz_block_int, 0, sizeof(int)*block_size*block_size);
                        memset(buffer_conflict_yz_block_int, 0, sizeof(int)*block_size*block_size);
                        memops_loop_time += omp_get_wtime() - time_start;
                        // printf("copying D[xb:%d, zb:%d ], D[yb:%d, zb:%d]\n", xb, zb, yb, zb);
                    }
                    // ystart = 0;
                    // zstart = 0;
                    // if(xb == yb && yb == zb){
                    //     xend = block_size - 1;
                    // }
                    #pragma omp task \
                    shared(n, block_size) \
                    firstprivate(xb, yb, zb) \
                    depend(in: D[yb + xb * n]) \
                    depend(in: D[zb + xb * n]) \
                    depend(in: D[zb + yb * n]) \
                    depend(inout: conflict_matrix_int[zb + xb * n]) \
                    depend(inout: conflict_matrix_int[zb + yb * n]) \
                    depend(inout: conflict_matrix_int[yb + xb * n]) \
                    depend(inout: task)
                    {
                        task++;
                        time_start = omp_get_wtime();
                        xend = (xb == yb && yb == zb) ? block_size - 1 : block_size;
                        for(x = 0; x < xend; ++x){
                            // if(xb == yb){
                            //     ystart = x + 1;
                            //     // conflict_yz_block += ystart*n;
                            // }
                            ystart = (xb == yb) ? x + 1 : 0;
                            if(xb == yb){
                                for(y = x + 1; y < block_size; ++y){
                                    // if(yb == zb){
                                    //     zstart = y + 1;
                                    // }
                                    // xy_reduction = 0.f;
                                    xy_reduction_int = 0;
                                    zstart = (yb == zb) ? y + 1 : 0;
                                    dist_xy = distance_xy_block[y + x * block_size];
                                    // contains_tie = 0.f;
                                    loop_len = block_size - zstart;
                                    if(yb == zb){
                                        // for (z = y + 1; z < block_size; ++z){
                                        #pragma unroll(16)
                                        for (z = 0; z < loop_len; ++z){
                                            idx = z + y + 1;
                                            //compute masks for conflict blocks.

                                            distance_check_1_mask = dist_xy < distance_xz_block[idx + x * block_size];
                                            distance_check_2_mask = dist_xy < distance_yz_block[idx + y * block_size];
                                            scalar_xy_closest_int = distance_check_1_mask & distance_check_2_mask;

                                            distance_check_1_mask = distance_xz_block[idx + x * block_size] < dist_xy;
                                            distance_check_2_mask =  distance_xz_block[idx + x * block_size] < distance_yz_block[idx + y * block_size];
                                            scalar_xz_closest_int = distance_check_1_mask & distance_check_2_mask;

                                            distance_check_1_mask = distance_yz_block[idx + y * block_size] < distance_xz_block[idx + x * block_size];
                                            distance_check_2_mask = distance_yz_block[idx + y * block_size] < dist_xy;
                                            scalar_yz_closest_int = distance_check_1_mask & distance_check_2_mask;


                                            xy_reduction_int += scalar_xz_closest_int + scalar_yz_closest_int;
                                            buffer_conflict_yz_block_int[idx + y * block_size] += scalar_xy_closest_int + scalar_xz_closest_int;
                                            buffer_conflict_xz_block_int[idx + x * block_size] += scalar_xy_closest_int + scalar_yz_closest_int;
                                        }

                                        buffer_conflict_xy_block_int[y + x * block_size] += xy_reduction_int;
                                    }
                                    else{
                                        // _mm512_store_epi32(buffer_conflict_xy_block_int + y + x * block_size, conflict_xy);
                                        // __m512 dist_xy_avx = _mm512_set1_ps(dist_xy);
                                        // __m512i all_ones = _mm512_set1_epi32(1);
                                        // __m512 dist_xz_avx, dist_yz_avx;
                                        // __mmask16 cmp_result_1, cmp_result_2, cmp_result_3;
                                        // __m512i conf_xy, conf_xz, conf_yz;
                                        // // for(z = 0; z < block_size; z+=16){
                                        // //     dist_xz_avx = _mm512_load_ps(distance_xz_block + z + x * block_size);
                                        // //     dist_yz_avx = _mm512_load_ps(distance_yz_block + z + y * block_size);
                                        // //     distance_check_1_mask = _mm512_cmplt_ps_mask(dist_xy_avx, dist_xz_avx);
                                        // //     distance_check_1_mask = _mm512_cmplt_ps_mask(dist_xy_avx, dist_yz_avx);
                                        // //     cmp_result_1 = distance_check_1_mask & distance_check_2_mask;

                                        // //     distance_check_1_mask = _mm512_cmplt_ps_mask(dist_xz_avx, dist_xy_avx);
                                        // //     distance_check_1_mask = _mm512_cmplt_ps_mask(dist_xz_avx, dist_yz_avx);
                                        // //     cmp_result_2 = distance_check_1_mask & distance_check_2_mask;

                                        // //     distance_check_1_mask = _mm512_cmplt_ps_mask(dist_yz_avx, dist_xy_avx);
                                        // //     distance_check_1_mask = _mm512_cmplt_ps_mask(dist_yz_avx, dist_xz_avx);
                                        // //     cmp_result_3 = distance_check_1_mask & distance_check_2_mask;

                                        // // }
                                        #pragma unroll(16)
                                        for (z = 0; z < block_size; ++z){
                                            //compute masks for conflict blocks.
                                            distance_check_1_mask = dist_xy < distance_xz_block[z + x * block_size];
                                            distance_check_2_mask = dist_xy < distance_yz_block[z + y * block_size];
                                            scalar_xy_closest_int = distance_check_1_mask & distance_check_2_mask;

                                            distance_check_1_mask = distance_xz_block[z + x * block_size] < dist_xy;
                                            distance_check_2_mask =  distance_xz_block[z + x * block_size] < distance_yz_block[z + y * block_size];
                                            scalar_xz_closest_int = distance_check_1_mask & distance_check_2_mask;

                                            distance_check_1_mask = distance_yz_block[z + y * block_size] < distance_xz_block[z + x * block_size];
                                            distance_check_2_mask = distance_yz_block[z + y * block_size] < dist_xy;
                                            scalar_yz_closest_int = distance_check_1_mask & distance_check_2_mask;

                                            xy_reduction_int += scalar_xz_closest_int + scalar_yz_closest_int;
                                            buffer_conflict_yz_block_int[z + y * block_size] += scalar_xy_closest_int + scalar_xz_closest_int;
                                            buffer_conflict_xz_block_int[z + x * block_size] += scalar_xy_closest_int + scalar_yz_closest_int;
                                        }
                                        buffer_conflict_xy_block_int[y + x * block_size] += xy_reduction_int;
                                    }
                                    // conflict_yz_block += n;
                                }
                            }
                            else{
                                for(y = 0; y < block_size; ++y){
                                    // if(yb == zb){
                                    //     zstart = y + 1;
                                    // }
                                    // xy_reduction = 0.f;
                                    xy_reduction_int = 0;
                                    zstart = (yb == zb) ? y + 1 : 0;
                                    dist_xy = distance_xy_block[y + x * block_size];
                                    // contains_tie = 0.f;
                                    loop_len = block_size - zstart;
                                    if(yb == zb){
                                        #pragma unroll(16)
                                        for (z = 0; z < loop_len; ++z){
                                            idx = z + y + 1;
                                            //compute masks for conflict blocks.

                                            distance_check_1_mask = dist_xy < distance_xz_block[idx + x * block_size];
                                            distance_check_2_mask = dist_xy < distance_yz_block[idx + y * block_size];
                                            scalar_xy_closest_int = distance_check_1_mask & distance_check_2_mask;

                                            distance_check_1_mask = distance_xz_block[idx + x * block_size] < dist_xy;
                                            distance_check_2_mask =  distance_xz_block[idx + x * block_size] < distance_yz_block[idx + y * block_size];
                                            scalar_xz_closest_int = distance_check_1_mask & distance_check_2_mask;

                                            distance_check_1_mask = distance_yz_block[idx + y * block_size] < distance_xz_block[idx + x * block_size];
                                            distance_check_2_mask = distance_yz_block[idx + y * block_size] < dist_xy;
                                            scalar_yz_closest_int = distance_check_1_mask & distance_check_2_mask;


                                            xy_reduction_int += scalar_xz_closest_int + scalar_yz_closest_int;
                                            buffer_conflict_yz_block_int[idx + y * block_size] += scalar_xy_closest_int + scalar_xz_closest_int;
                                            buffer_conflict_xz_block_int[idx + x * block_size] += scalar_xy_closest_int + scalar_yz_closest_int;
                                        }
                                        buffer_conflict_xy_block_int[y + x * block_size] += xy_reduction_int;
                                    }
                                    else{
                                        #pragma unroll(16)
                                        for (z = 0; z < block_size; ++z){
                                            //compute masks for conflict blocks.
                                            distance_check_1_mask = dist_xy < distance_xz_block[z + x * block_size];
                                            distance_check_2_mask = dist_xy < distance_yz_block[z + y * block_size];
                                            scalar_xy_closest_int = distance_check_1_mask & distance_check_2_mask;

                                            distance_check_1_mask = distance_xz_block[z + x * block_size] < dist_xy;
                                            distance_check_2_mask =  distance_xz_block[z + x * block_size] < distance_yz_block[z + y * block_size];
                                            scalar_xz_closest_int = distance_check_1_mask & distance_check_2_mask;

                                            distance_check_1_mask = distance_yz_block[z + y * block_size] < distance_xz_block[z + x * block_size];
                                            distance_check_2_mask = distance_yz_block[z + y * block_size] < dist_xy;
                                            scalar_yz_closest_int = distance_check_1_mask & distance_check_2_mask;
                                            xy_reduction_int += scalar_xz_closest_int + scalar_yz_closest_int;
                                            buffer_conflict_yz_block_int[z + y * block_size] += scalar_xy_closest_int + scalar_xz_closest_int;
                                            buffer_conflict_xz_block_int[z + x * block_size] += scalar_xy_closest_int + scalar_yz_closest_int;
                                        }
                                        buffer_conflict_xy_block_int[y + x * block_size] += xy_reduction_int;
                                    }
                                    // conflict_yz_block += n;
                                }
                            }

                        }
                        conflict_loop_time += omp_get_wtime() - time_start;
                    }
                    // conflict_xy_block = conflict_matrix + yb + xb * n;
                    // conflict_xz_block = conflict_matrix + zb + xb * n;
                    // conflict_yz_block = conflict_matrix + zb + yb * n;
                    #pragma omp task \
                    shared(n, block_size, conflict_matrix_int) \
                    firstprivate(xb, zb) \
                    depend(inout: conflict_matrix_int[zb + xb * n]) \
                    depend(inout: task)
                    {
                        task++;
                        time_start2 = omp_get_wtime();
                        conflict_xz_block_int = conflict_matrix_int + zb + xb * n;
                        for(i = 0; i < block_size; ++i){
                            for(j = 0; j < block_size; ++j){
                                conflict_xz_block_int[j + i * n] += buffer_conflict_xz_block_int[j + i * block_size];
                            }
                        }
                        memops_loop_time += omp_get_wtime() - time_start2;
                    }
                    #pragma omp task \
                    shared(n, block_size, conflict_matrix_int) \
                    firstprivate(yb, zb) \
                    depend(inout: conflict_matrix_int[zb + yb * n]) \
                    depend(inout: task)
                    {
                        task++;
                        time_start2 = omp_get_wtime();
                        conflict_yz_block_int = conflict_matrix_int + zb + yb * n;
                        for(i = 0; i < block_size; ++i){
                            for(j = 0; j < block_size; ++j){
                                conflict_yz_block_int[j + i * n] += buffer_conflict_yz_block_int[j + i * block_size];
                            }

                        }
                        printf("writing Conf[xb:%d, zb:%d ], Conf[yb:%d, zb:%d]\n", xb, zb, yb, zb);
                        print_matrix_int(block_size, n, conflict_xz_block_int);
                        print_matrix_int(block_size, n, conflict_yz_block_int);
                        memops_loop_time += omp_get_wtime() - time_start2;
                    }

                }
                #pragma omp task \
                shared(n, block_size, conflict_matrix_int) \
                firstprivate(xb, yb) \
                depend(inout: conflict_matrix_int[yb + xb * n]) \
                depend(inout: task)
                {
                    task++;
                // conflict_xy_block = conflict_matrix + yb + xb * n;
                    time_start2 = omp_get_wtime();
                    conflict_xy_block_int = conflict_matrix_int + yb + xb * n;
                    for(i = 0; i < block_size; ++i){
                        for(j = 0; j < block_size; ++j){
                            // conflict_xy_block[j + i * n] += buffer_conflict_xy_block[j + i * block_size];
                            conflict_xy_block_int[j + i * n] += buffer_conflict_xy_block_int[j + i * block_size];
                        }
                        // conflict_xy_block += n;
                    }
                    printf("writing Conf[xb:%d, yb:%d ]\n", xb, yb);
                    print_matrix_int(block_size, n, conflict_xy_block_int);
                    memops_loop_time += omp_get_wtime() - time_start2;
                }
            }
        }
        time_start = omp_get_wtime();
        #pragma omp parallel for shared(conflict_matrix, conflict_matrix_int, n)
        for(i = 0; i < n * n; ++i){
            // conflict_matrix[i] = 1.f/conflict_matrix[i];
            conflict_matrix[i] = 1.f/conflict_matrix_int[i];
        }
        print_matrix_int(n, n, conflict_matrix_int);
        conflict_loop_time += omp_get_wtime() - time_start;
        // return;
        // printf("\n\n");
            // initialize diagonal of C.
        float sum;
        time_start = omp_get_wtime();
        #pragma omp parallel for private(i, j, sum) shared(C, conflict_matrix, n)
        for (i = 0; i < n; ++i){
            sum = 0.f;
            for (j = 0; j < i; ++j){
                sum += conflict_matrix[i + j * n];
            }
            for (j = i + 1; j < n; ++j){
                sum += conflict_matrix[j + i * n];
            }
            C[i + i * n] = sum;
        }
        cohesion_loop_time += omp_get_wtime() - time_start;
        iters = 0;
        time_start = omp_get_wtime();
        _mm_free(conflict_matrix_int);
        _mm_free(buffer_conflict_xy_block_int); _mm_free(buffer_conflict_xz_block_int); _mm_free(buffer_conflict_yz_block_int);
        _mm_free(distance_xy_block); _mm_free(distance_xz_block); _mm_free(distance_yz_block);
        memops_loop_time += omp_get_wtime() - time_start;
    }
    return;
        double time_start = 0.0, time_start2 = 0.0;
        time_start = omp_get_wtime();
        double memops_loop_time = 0.0, conflict_loop_time = 0.0, cohesion_loop_time = 0.0;
        int i, j, k, x, y, z, idx, zstart, ystart, xend, loop_len;
        int xb, yb, zb;
        unsigned int distance_check_1_mask, distance_check_2_mask;
        float dist_xy = 0.f, conflict_xy_val = 0.f;
        float xy_reduction = 0.f, yx_reduction = 0.f, cohesion_sum = 0.f;
        float *conflict_xy_block, *conflict_xz_block, *conflict_yz_block;
        float *cohesion_xy_block ;
        float *cohesion_yx_block;
        float *cohesion_xz_block;
        float *cohesion_zx_block;
        float *cohesion_yz_block;
        float *cohesion_zy_block;
        block_size /= 2;

        // float* restrict mask_xy_closest = (float *) _mm_malloc(block_size * sizeof(float), VECALIGN);
        // float* restrict mask_xz_closest = (float *) _mm_malloc(block_size * sizeof(float), VECALIGN);
        // float* restrict mask_yz_closest = (float *) _mm_malloc(block_size * sizeof(float), VECALIGN);

        float scalar_xy_closest, scalar_xz_closest, scalar_yz_closest;

        // float* restrict mask_tie_xy_xz = (float *) _mm_malloc(block_size * sizeof(float), VECALIGN);
        // float* restrict mask_tie_xy_yz = (float *) _mm_malloc(block_size * sizeof(float), VECALIGN);
        // float* restrict mask_tie_xz_yz = (float *) _mm_malloc(block_size * sizeof(float), VECALIGN);

        float* buffer_conflict_xz_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
        float* buffer_conflict_yz_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
        // float* buffer_conflict_xy_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);

        float* restrict buffer_zx_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
        float* restrict buffer_zy_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
        float* restrict buffer_xz_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
        float* restrict buffer_yz_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
        float* restrict buffer_xy_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
        float* restrict buffer_yx_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);

        float* distance_xy_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
        float* distance_xz_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);
        float* distance_yz_block = (float *) _mm_malloc(block_size * block_size * sizeof(float), VECALIGN);

        float* tmp_buffer_conflict_yz_block = buffer_conflict_yz_block;
        float* tmp_distance_yz_block = distance_yz_block;
        float* tmp_buffer_yz_block = buffer_yz_block;
        float* tmp_buffer_zy_block = buffer_zy_block;
        memops_loop_time += omp_get_wtime() - time_start;

        for(xb = 0; xb < n; xb += block_size){
            for(yb = xb; yb < n; yb += block_size){
                time_start = omp_get_wtime();
                // conflict_xy_block = conflict_matrix + yb + xb * n;
                for (i = 0; i < block_size; ++i){
                    memcpy(distance_xy_block + i * block_size, D + yb + (xb + i) * n, sizeof(float)*block_size);
                    // memcpy(buffer_conflict_xy_block + i * block_size, conflict_xy_block + i * n, sizeof(float)*block_size);
                }

                memset(buffer_yx_block,0,sizeof(float)*block_size*block_size);
                memset(buffer_xy_block,0,sizeof(float)*block_size*block_size);
                memops_loop_time += omp_get_wtime() - time_start;
                for(zb = yb; zb < n; zb += block_size){
                    time_start = omp_get_wtime();
                    conflict_xy_block = conflict_matrix + yb + xb * n;
                    conflict_xz_block = conflict_matrix + zb + xb * n;
                    conflict_yz_block = conflict_matrix + zb + yb * n;
                    if(xb == yb){
                        #pragma unroll(8)
                        for (i = 0; i < block_size; ++i){
                            memcpy(distance_xz_block + i * block_size, D + zb + (xb + i) * n, sizeof(float)*block_size);
                            memcpy(buffer_conflict_xz_block + i * block_size, conflict_xz_block + i * n, sizeof(float)*block_size);
                            // memcpy(distance_yz_block + i * block_size, D + zb + (yb + i) * n, sizeof(float)*block_size);
                            // memcpy(buffer_conflict_yz_block + i * block_size, conflict_yz_block + i * n, sizeof(float)*block_size);
                        }
                        distance_yz_block = distance_xz_block;
                        buffer_conflict_yz_block = buffer_conflict_xz_block;
                        memset(buffer_zx_block,0,sizeof(float)*block_size*block_size);
                        memset(buffer_xz_block,0,sizeof(float)*block_size*block_size);

                        buffer_zy_block = buffer_zx_block;
                        buffer_yz_block = buffer_xz_block;
                    }
                    else{
                        distance_yz_block = tmp_distance_yz_block;
                        buffer_conflict_yz_block = tmp_buffer_conflict_yz_block;
                        buffer_zy_block = tmp_buffer_zy_block;
                        buffer_yz_block = tmp_buffer_yz_block;
                        #pragma unroll(8)
                        for (i = 0; i < block_size; ++i){
                            memcpy(distance_xz_block + i * block_size, D + zb + (xb + i) * n, sizeof(float)*block_size);
                            memcpy(buffer_conflict_xz_block + i * block_size, conflict_xz_block + i * n, sizeof(float)*block_size);
                            memcpy(distance_yz_block + i * block_size, D + zb + (yb + i) * n, sizeof(float)*block_size);
                            memcpy(buffer_conflict_yz_block + i * block_size, conflict_yz_block + i * n, sizeof(float)*block_size);
                        }
                        memset(buffer_zx_block,0,sizeof(float)*block_size*block_size);
                        memset(buffer_zy_block,0,sizeof(float)*block_size*block_size);
                        memset(buffer_xz_block,0,sizeof(float)*block_size*block_size);
                        memset(buffer_yz_block,0,sizeof(float)*block_size*block_size);
                    }
                    memops_loop_time += omp_get_wtime() - time_start;

                    time_start = omp_get_wtime();
                    xend = block_size;
                    // ystart = 0;
                    // zstart = 0;
                    if(xb == yb && yb == zb){
                        xend = block_size - 1;
                    }
                    for(x = 0; x < xend; ++x){
                        // if(xb == yb){
                        //     ystart = x + 1;

                        // }
                        ystart = (xb == yb) ? x + 1 : 0;
                        for(y = ystart; y < block_size; ++y){
                            xy_reduction = 0.f; yx_reduction = 0.f;
                            // if(yb == zb){
                            //     zstart = y + 1;
                            // }
                                // zstart = (yb == zb) ? y + 1 : 0;
                            dist_xy = distance_xy_block[y + x * block_size];
                            // loop_len = block_size - zstart;
                            if(yb == zb){
                                loop_len = block_size - y - 1;
                                conflict_xy_val = conflict_xy_block[y];
                                // for (z = y + 1; z < block_size; ++z){
                                for (z = 0; z < loop_len; ++z){
                                    //compute masks for conflict blocks.
                                    idx = z + y + 1;
                                    distance_check_1_mask = dist_xy < distance_xz_block[idx + x * block_size];
                                    distance_check_2_mask = dist_xy < distance_yz_block[idx + y * block_size];
                                    scalar_xy_closest = distance_check_1_mask & distance_check_2_mask;

                                    distance_check_1_mask = distance_xz_block[idx + x * block_size] < dist_xy;
                                    distance_check_2_mask =  distance_xz_block[idx + x * block_size] < distance_yz_block[idx + y * block_size];
                                    scalar_xz_closest = distance_check_1_mask & distance_check_2_mask;

                                    distance_check_1_mask = distance_yz_block[idx + y * block_size] < distance_xz_block[idx + x * block_size];
                                    distance_check_2_mask = distance_yz_block[idx + y * block_size] < dist_xy;
                                    scalar_yz_closest = distance_check_1_mask & distance_check_2_mask;

                                    // xy closest pair.
                                    yx_reduction += scalar_xy_closest*buffer_conflict_xz_block[idx + x * block_size];
                                    xy_reduction += scalar_xy_closest*buffer_conflict_yz_block[idx + y * block_size];

                                    // xz closest pair.
                                    buffer_xz_block[idx + x * block_size] += scalar_xz_closest*buffer_conflict_yz_block[idx + y * block_size];
                                    buffer_zx_block[idx + x * block_size] += scalar_xz_closest*conflict_xy_val;

                                    // yz closest pair.
                                    buffer_yz_block[idx + y * block_size] += scalar_yz_closest*buffer_conflict_xz_block[idx + x * block_size];
                                    buffer_zy_block[idx + y * block_size] += scalar_yz_closest*conflict_xy_val;
                                }
                                buffer_xy_block[y + x * block_size] += xy_reduction;
                                buffer_yx_block[y + x * block_size] += yx_reduction;
                            }
                            else{
                                conflict_xy_val = conflict_xy_block[y];
                                // __m512 all_ones = _mm512_set1_ps(1.f);
                                // __m512 dist_xy_avx = _mm512_set1_ps(dist_xy);
                                // __m512 conflict_xy_avx = _mm512_set1_ps(conflict_xy_val);

                                // __m512 xy_reduction_avx = _mm512_setzero();
                                // __m512 yx_reduction_avx = _mm512_setzero();
                                // __m512 dist_xz_avx, dist_yz_avx, conflict_xz_avx, conflict_yz_avx;
                                // __m512 cohesion_xz_avx, cohesion_zx_avx, cohesion_yz_avx, cohesion_zy_avx;
                                // __mmask16 mask_xy_closest, mask_xz_closest, mask_yz_closest;

                                // __m512 xy_reduction_avx_1 = _mm512_setzero();
                                // __m512 yx_reduction_avx_1 = _mm512_setzero();
                                // __m512 dist_xz_avx_1, dist_yz_avx_1, conflict_xz_avx_1, conflict_yz_avx_1;
                                // __m512 cohesion_xz_avx_1, cohesion_zx_avx_1, cohesion_yz_avx_1, cohesion_zy_avx_1;
                                // __mmask16 mask_xy_closest_1, mask_xz_closest_1, mask_yz_closest_1;

                                // __m512 xy_reduction_avx_2 = _mm512_setzero();
                                // __m512 yx_reduction_avx_2 = _mm512_setzero();
                                // __m512 dist_xz_avx_2, dist_yz_avx_2, conflict_xz_avx_2, conflict_yz_avx_2;
                                // __m512 cohesion_xz_avx_2, cohesion_zx_avx_2, cohesion_yz_avx_2, cohesion_zy_avx_2;
                                // __mmask16 mask_xy_closest_2, mask_xz_closest_2, mask_yz_closest_2;

                                // __m512 xy_reduction_avx_3 = _mm512_setzero();
                                // __m512 yx_reduction_avx_3 = _mm512_setzero();
                                // __m512 dist_xz_avx_3, dist_yz_avx_3, conflict_xz_avx_3, conflict_yz_avx_3;
                                // __m512 cohesion_xz_avx_3, cohesion_zx_avx_3, cohesion_yz_avx_3, cohesion_zy_avx_3;
                                // __mmask16 mask_xy_closest_3, mask_xz_closest_3, mask_yz_closest_3;

                                // __mmask16 distance_check_3_mask, distance_check_4_mask;
                                // __mmask16 distance_check_5_mask, distance_check_6_mask;
                                // __mmask16 distance_check_7_mask, distance_check_8_mask;

                                // for(z = 0; z < block_size; z += 128){
                                //     dist_xz_avx = _mm512_load_ps(distance_xz_block + z + x * block_size);
                                //     dist_yz_avx = _mm512_load_ps(distance_yz_block + z + y * block_size);
                                //     cohesion_xz_avx = _mm512_load_ps(buffer_xz_block + z + x * block_size);
                                //     cohesion_zx_avx = _mm512_load_ps(buffer_zx_block + z + x * block_size);
                                //     cohesion_yz_avx = _mm512_load_ps(buffer_yz_block + z + y * block_size);
                                //     cohesion_zy_avx = _mm512_load_ps(buffer_zy_block + z + y * block_size);
                                //     conflict_yz_avx = _mm512_load_ps(buffer_conflict_yz_block + z + y * block_size);
                                //     conflict_xz_avx = _mm512_load_ps(buffer_conflict_xz_block + z + x * block_size);

                                //     dist_xz_avx_1 = _mm512_load_ps(distance_xz_block + z + 16 + x * block_size);
                                //     dist_yz_avx_1 = _mm512_load_ps(distance_yz_block + z + 16 + y * block_size);
                                //     cohesion_xz_avx_1 = _mm512_load_ps(buffer_xz_block + z + 16 + x * block_size);
                                //     cohesion_zx_avx_1 = _mm512_load_ps(buffer_zx_block + z + 16 + x * block_size);
                                //     cohesion_yz_avx_1 = _mm512_load_ps(buffer_yz_block + z + 16 + y * block_size);
                                //     cohesion_zy_avx_1 = _mm512_load_ps(buffer_zy_block + z + 16 + y * block_size);
                                //     conflict_yz_avx_1 = _mm512_load_ps(buffer_conflict_yz_block + z + 16 + y * block_size);
                                //     conflict_xz_avx_1 = _mm512_load_ps(buffer_conflict_xz_block + z + 16 + x * block_size);

                                //     dist_xz_avx_2 = _mm512_load_ps(distance_xz_block + z + 32 + x * block_size);
                                //     dist_yz_avx_2 = _mm512_load_ps(distance_yz_block + z + 32 + y * block_size);
                                //     cohesion_xz_avx_2 = _mm512_load_ps(buffer_xz_block + z + 32 + x * block_size);
                                //     cohesion_zx_avx_2 = _mm512_load_ps(buffer_zx_block + z + 32 + x * block_size);
                                //     cohesion_yz_avx_2 = _mm512_load_ps(buffer_yz_block + z + 32 + y * block_size);
                                //     cohesion_zy_avx_2 = _mm512_load_ps(buffer_zy_block + z + 32 + y * block_size);
                                //     conflict_yz_avx_2 = _mm512_load_ps(buffer_conflict_yz_block + z + 32 + y * block_size);
                                //     conflict_xz_avx_2 = _mm512_load_ps(buffer_conflict_xz_block + z + 32 + x * block_size);

                                //     dist_xz_avx_3 = _mm512_load_ps(distance_xz_block + z + 64 + x * block_size);
                                //     dist_yz_avx_3 = _mm512_load_ps(distance_yz_block + z + 64 + y * block_size);
                                //     cohesion_xz_avx_3 = _mm512_load_ps(buffer_xz_block + z + 64 + x * block_size);
                                //     cohesion_zx_avx_3 = _mm512_load_ps(buffer_zx_block + z + 64 + x * block_size);
                                //     cohesion_yz_avx_3 = _mm512_load_ps(buffer_yz_block + z + 64 + y * block_size);
                                //     cohesion_zy_avx_3 = _mm512_load_ps(buffer_zy_block + z + 64 + y * block_size);
                                //     conflict_yz_avx_3 = _mm512_load_ps(buffer_conflict_yz_block + z + 64 + y * block_size);
                                //     conflict_xz_avx_3 = _mm512_load_ps(buffer_conflict_xz_block + z + 64 + x * block_size);

                                //     distance_check_1_mask = _mm512_cmplt_ps_mask(dist_xy_avx, dist_xz_avx);
                                //     distance_check_2_mask = _mm512_cmplt_ps_mask(dist_xy_avx, dist_yz_avx);
                                //     mask_xy_closest = distance_check_1_mask & distance_check_2_mask;

                                //     distance_check_1_mask = _mm512_cmplt_ps_mask(dist_xz_avx, dist_xy_avx);
                                //     distance_check_2_mask = _mm512_cmplt_ps_mask(dist_xz_avx, dist_yz_avx);
                                //     mask_xz_closest = distance_check_1_mask & distance_check_2_mask;

                                //     distance_check_1_mask = _mm512_cmplt_ps_mask(dist_yz_avx, dist_xz_avx);
                                //     distance_check_2_mask = _mm512_cmplt_ps_mask(dist_yz_avx, dist_xy_avx);
                                //     mask_yz_closest = distance_check_1_mask & distance_check_2_mask;

                                //     distance_check_3_mask = _mm512_cmplt_ps_mask(dist_xy_avx, dist_xz_avx);
                                //     distance_check_4_mask = _mm512_cmplt_ps_mask(dist_xy_avx, dist_yz_avx);
                                //     mask_xy_closest_1 = distance_check_3_mask & distance_check_4_mask;

                                //     distance_check_3_mask = _mm512_cmplt_ps_mask(dist_xz_avx, dist_xy_avx);
                                //     distance_check_4_mask = _mm512_cmplt_ps_mask(dist_xz_avx, dist_yz_avx);
                                //     mask_xz_closest_1 = distance_check_3_mask & distance_check_4_mask;

                                //     distance_check_3_mask = _mm512_cmplt_ps_mask(dist_yz_avx, dist_xz_avx);
                                //     distance_check_4_mask = _mm512_cmplt_ps_mask(dist_yz_avx, dist_xy_avx);
                                //     mask_yz_closest_1 = distance_check_3_mask & distance_check_4_mask;

                                //     distance_check_5_mask = _mm512_cmplt_ps_mask(dist_xy_avx, dist_xz_avx);
                                //     distance_check_6_mask = _mm512_cmplt_ps_mask(dist_xy_avx, dist_yz_avx);
                                //     mask_xy_closest_2 = distance_check_5_mask & distance_check_6_mask;

                                //     distance_check_5_mask = _mm512_cmplt_ps_mask(dist_xz_avx, dist_xy_avx);
                                //     distance_check_6_mask = _mm512_cmplt_ps_mask(dist_xz_avx, dist_yz_avx);
                                //     mask_xz_closest_2 = distance_check_5_mask & distance_check_6_mask;

                                //     distance_check_5_mask = _mm512_cmplt_ps_mask(dist_yz_avx, dist_xz_avx);
                                //     distance_check_6_mask = _mm512_cmplt_ps_mask(dist_yz_avx, dist_xy_avx);
                                //     mask_yz_closest_2 = distance_check_5_mask & distance_check_6_mask;

                                //     distance_check_7_mask = _mm512_cmplt_ps_mask(dist_xy_avx, dist_xz_avx);
                                //     distance_check_8_mask = _mm512_cmplt_ps_mask(dist_xy_avx, dist_yz_avx);
                                //     mask_xy_closest_3 = distance_check_7_mask & distance_check_8_mask;

                                //     distance_check_7_mask = _mm512_cmplt_ps_mask(dist_xz_avx, dist_xy_avx);
                                //     distance_check_8_mask = _mm512_cmplt_ps_mask(dist_xz_avx, dist_yz_avx);
                                //     mask_xz_closest_3 = distance_check_7_mask & distance_check_8_mask;

                                //     distance_check_7_mask = _mm512_cmplt_ps_mask(dist_yz_avx, dist_xz_avx);
                                //     distance_check_8_mask = _mm512_cmplt_ps_mask(dist_yz_avx, dist_xy_avx);
                                //     mask_yz_closest_3 = distance_check_7_mask & distance_check_8_mask;


                                //     xy_reduction_avx = _mm512_mask_fmadd_ps(all_ones, mask_xy_closest, conflict_yz_avx, xy_reduction_avx);
                                //     yx_reduction_avx = _mm512_mask_fmadd_ps(all_ones, mask_xy_closest, conflict_xz_avx, yx_reduction_avx);

                                //     xy_reduction_avx_1 = _mm512_mask_fmadd_ps(all_ones, mask_xy_closest_1, conflict_yz_avx_1, xy_reduction_avx_1);
                                //     yx_reduction_avx_1 = _mm512_mask_fmadd_ps(all_ones, mask_xy_closest_1, conflict_xz_avx_1, yx_reduction_avx_1);

                                //     xy_reduction_avx_2 = _mm512_mask_fmadd_ps(all_ones, mask_xy_closest_2, conflict_yz_avx_2, xy_reduction_avx_2);
                                //     yx_reduction_avx_2 = _mm512_mask_fmadd_ps(all_ones, mask_xy_closest_2, conflict_xz_avx_2, yx_reduction_avx_2);

                                //     xy_reduction_avx_3 = _mm512_mask_fmadd_ps(all_ones, mask_xy_closest_3, conflict_yz_avx_3, xy_reduction_avx_3);
                                //     yx_reduction_avx_3 = _mm512_mask_fmadd_ps(all_ones, mask_xy_closest_3, conflict_xz_avx_3, yx_reduction_avx_3);

                                //     _mm512_store_ps(buffer_xz_block + z + x * block_size, _mm512_mask_fmadd_ps(all_ones, mask_xz_closest, conflict_yz_avx, cohesion_xz_avx));
                                //     _mm512_store_ps(buffer_zx_block + z + x * block_size, _mm512_mask_fmadd_ps(all_ones, mask_xz_closest, conflict_xy_avx, cohesion_zx_avx));
                                //     _mm512_store_ps(buffer_yz_block + z + y * block_size, _mm512_mask_fmadd_ps(all_ones, mask_yz_closest, conflict_xz_avx, cohesion_yz_avx));
                                //     _mm512_store_ps(buffer_zy_block + z + y * block_size, _mm512_mask_fmadd_ps(all_ones, mask_yz_closest, conflict_xy_avx, cohesion_zy_avx));

                                //     _mm512_store_ps(buffer_xz_block + z + 16 + x * block_size, _mm512_mask_fmadd_ps(all_ones, mask_xz_closest_1, conflict_yz_avx_1, cohesion_xz_avx_1));
                                //     _mm512_store_ps(buffer_zx_block + z + 16 + x * block_size, _mm512_mask_fmadd_ps(all_ones, mask_xz_closest_1, conflict_xy_avx, cohesion_zx_avx_1));
                                //     _mm512_store_ps(buffer_yz_block + z + 16 + y * block_size, _mm512_mask_fmadd_ps(all_ones, mask_yz_closest_1, conflict_xz_avx_1, cohesion_yz_avx_1));
                                //     _mm512_store_ps(buffer_zy_block + z + 16 + y * block_size, _mm512_mask_fmadd_ps(all_ones, mask_yz_closest_1, conflict_xy_avx, cohesion_zy_avx_1));

                                //     _mm512_store_ps(buffer_xz_block + z + 32 + x * block_size, _mm512_mask_fmadd_ps(all_ones, mask_xz_closest_2, conflict_yz_avx_2, cohesion_xz_avx_2));
                                //     _mm512_store_ps(buffer_zx_block + z + 32 + x * block_size, _mm512_mask_fmadd_ps(all_ones, mask_xz_closest_2, conflict_xy_avx, cohesion_zx_avx_2));
                                //     _mm512_store_ps(buffer_yz_block + z + 32 + y * block_size, _mm512_mask_fmadd_ps(all_ones, mask_yz_closest_2, conflict_xz_avx_2, cohesion_yz_avx_2));
                                //     _mm512_store_ps(buffer_zy_block + z + 32 + y * block_size, _mm512_mask_fmadd_ps(all_ones, mask_yz_closest_2, conflict_xy_avx, cohesion_zy_avx_2));

                                //     _mm512_store_ps(buffer_xz_block + z + 64 + x * block_size, _mm512_mask_fmadd_ps(all_ones, mask_xz_closest_3, conflict_yz_avx_3, cohesion_xz_avx_3));
                                //     _mm512_store_ps(buffer_zx_block + z + 64 + x * block_size, _mm512_mask_fmadd_ps(all_ones, mask_xz_closest_3, conflict_xy_avx, cohesion_zx_avx_3));
                                //     _mm512_store_ps(buffer_yz_block + z + 64 + y * block_size, _mm512_mask_fmadd_ps(all_ones, mask_yz_closest_3, conflict_xz_avx_3, cohesion_yz_avx_3));
                                //     _mm512_store_ps(buffer_zy_block + z + 64 + y * block_size, _mm512_mask_fmadd_ps(all_ones, mask_yz_closest_3, conflict_xy_avx, cohesion_zy_avx_3));
                                // }
                                // xy_reduction_avx += xy_reduction_avx_1 + xy_reduction_avx_2 + xy_reduction_avx_3;
                                // yx_reduction_avx += yx_reduction_avx_1 + yx_reduction_avx_2 + yx_reduction_avx_3;
                                // xy_reduction += _mm512_reduce_add_ps(xy_reduction_avx);
                                // yx_reduction += _mm512_reduce_add_ps(yx_reduction_avx);
                                #pragma unroll(8)
                                //update cohesion blocks.
                                for (z = 0; z < block_size; ++z){
                                    // xy closest pair.
                                    distance_check_1_mask = dist_xy < distance_xz_block[z + x * block_size];
                                    distance_check_2_mask = dist_xy < distance_yz_block[z + y * block_size];
                                    scalar_xy_closest = distance_check_1_mask & distance_check_2_mask;
                                    xy_reduction += scalar_xy_closest*buffer_conflict_yz_block[z + y * block_size];
                                    yx_reduction += scalar_xy_closest*buffer_conflict_xz_block[z + x * block_size];

                                    // xz closest pair.
                                    distance_check_1_mask = distance_xz_block[z + x * block_size] < dist_xy;
                                    distance_check_2_mask =  distance_xz_block[z + x * block_size] < distance_yz_block[z + y * block_size];
                                    scalar_xz_closest = distance_check_1_mask & distance_check_2_mask;
                                    buffer_xz_block[z + x * block_size] += scalar_xz_closest*buffer_conflict_yz_block[z + y * block_size];
                                    buffer_zx_block[z + x * block_size] += scalar_xz_closest*conflict_xy_val;

                                    //yz closest pair.
                                    distance_check_1_mask = distance_yz_block[z + y * block_size] < distance_xz_block[z + x * block_size];
                                    distance_check_2_mask = distance_yz_block[z + y * block_size] < dist_xy;
                                    scalar_yz_closest = distance_check_1_mask & distance_check_2_mask;
                                    buffer_yz_block[z + y * block_size] += scalar_yz_closest*buffer_conflict_xz_block[z + x * block_size];
                                    buffer_zy_block[z + y * block_size] += scalar_yz_closest*conflict_xy_val;
                                }
                                buffer_xy_block[y + x * block_size] += xy_reduction;
                                buffer_yx_block[y + x * block_size] += yx_reduction;
                            }
                        }
                        conflict_xy_block += n;
                    }


                    time_start2 = omp_get_wtime();
                    if(xb == yb){
                        cohesion_zx_block = C + xb + zb * n;
                        cohesion_xz_block = C + zb + xb * n;
                        for(i = 0; i < block_size; ++i){
                            for(j = 0; j < block_size; ++j){
                                // printf("idx: %d\n", n*j + i);
                                cohesion_zx_block[j] += buffer_zx_block[i + j * block_size];
                                // cohesion_yx_block[j] += buffer_yx_block[i + j * block_size];
                            }
                            cohesion_zx_block += n;
                        }
                        for(i = 0; i < block_size; ++i){
                            for(j = 0; j < block_size; ++j){
                                cohesion_xz_block[j] += buffer_xz_block[j + i * block_size];
                                // cohesion_xy_block[j] += buffer_xy_block[j + i * block_size];
                            }
                            cohesion_xz_block += n;
                        }

                    }
                    else{
                        cohesion_zx_block = C + xb + zb * n;
                        cohesion_zy_block = C + yb + zb * n;
                        cohesion_xz_block = C + zb + xb * n;
                        cohesion_yz_block = C + zb + yb * n;
                        for(i = 0; i < block_size; ++i){
                            for(j = 0; j < block_size; ++j){
                                // printf("idx: %d\n", n*j + i);
                                cohesion_zx_block[j] += buffer_zx_block[i + j * block_size];
                                cohesion_zy_block[j] += buffer_zy_block[i + j * block_size];
                                // cohesion_yx_block[j] += buffer_yx_block[i + j * block_size];
                            }
                            cohesion_zx_block += n;
                            cohesion_zy_block += n;
                        }
                        for(i = 0; i < block_size; ++i){
                            for(j = 0; j < block_size; ++j){
                                cohesion_xz_block[j] += buffer_xz_block[j + i * block_size];
                                cohesion_yz_block[j] += buffer_yz_block[j + i * block_size];
                                // cohesion_xy_block[j] += buffer_xy_block[j + i * block_size];
                            }
                            cohesion_xz_block += n;
                            cohesion_yz_block += n;
                        }
                    }

                    memops_loop_time += omp_get_wtime() - time_start2;
                    cohesion_loop_time += omp_get_wtime() - time_start;
                }
                time_start2 = omp_get_wtime();
                cohesion_xy_block = C + yb + xb * n;
                cohesion_yx_block = C + xb + yb * n;
                for(i = 0; i < block_size; ++i){
                    for(j = 0; j < block_size; ++j){
                        // printf("idx: %d\n", n*j + i);
                        cohesion_yx_block[j] += buffer_yx_block[i + j * block_size];
                    }
                    cohesion_yx_block += n;
                }
                cohesion_yx_block = C + xb + yb * n;

                conflict_xy_block = conflict_matrix + yb + xb * n;

                for(i = 0; i < block_size; ++i){
                    for(j = 0; j < block_size; ++j){
                        cohesion_xy_block[j] += buffer_xy_block[j + i * block_size];
                    }
                    cohesion_xy_block += n;
                }
                memops_loop_time += omp_get_wtime() - time_start2;
            }
        }
        // print_matrix(n, n, C);

        printf("==============================================\n");
        printf("Seq. Triplet Intrinsics OMP Loop Times\n");
        printf("==============================================\n");

        printf("memops loop time: %.5fs\n", memops_loop_time);
        printf("conflict focus size loop time: %.5fs\n", conflict_loop_time);
        printf("cohesion matrix update loop time: %.5fs\n\n", cohesion_loop_time);

        _mm_free(distance_xy_block); _mm_free(distance_xz_block); _mm_free(tmp_distance_yz_block);
        // _mm_free(mask_tie_xy_xz); _mm_free(mask_tie_xy_yz); _mm_free(mask_tie_xz_yz);
        _mm_free(buffer_zx_block); _mm_free(tmp_buffer_zy_block); _mm_free(buffer_yx_block);
        _mm_free(buffer_xz_block); _mm_free(tmp_buffer_yz_block); _mm_free(buffer_xy_block);
        // _mm_free(mask_xy_closest); _mm_free(mask_xz_closest); _mm_free(mask_yz_closest);
        _mm_free(buffer_conflict_xz_block); _mm_free(tmp_buffer_conflict_yz_block);
        _mm_free(conflict_matrix);

        // _mm_free(buffer_conflict_xy_block);
        // _mm_free(conflict_matrix_int);
        // _mm_free(mask_xy_closest_int); _mm_free(mask_xz_closest_int); _mm_free(mask_yz_closest_int);
        // _mm_free(buffer_conflict_xz_block_int); _mm_free(buffer_conflict_yz_block_int); _mm_free(buffer_conflict_xy_block_int);

}
