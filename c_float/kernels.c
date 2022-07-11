#include "kernels.h"
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
                        // z equidistant to x and y
                        C[lin(x, z, n)] += 0.5f / cfs;
                        C[lin(y, z, n)] += 0.5f / cfs;
                    }
                }
            }
        }
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
    // declare indices
    int x, y, z, i, j, k, xb, yb, ib;
    // pre-allocate buffers for conflict focus and distance blocks
    float * conflict_block = (float *) _mm_malloc(block_size * block_size * sizeof(float),VECALIGN);
    float * distance_block = (float *) _mm_malloc(block_size * block_size * sizeof(float),VECALIGN);
    // pre-allocate mask buffers for z in conflict focus, z support x and z supports both.
    float * mask_z_in_conflict_focus = (float *) _mm_malloc(block_size * sizeof(float),VECALIGN);
    float * mask_z_supports_x = (float *) _mm_malloc(block_size  * sizeof(float),VECALIGN);
    float * mask_z_supports_x_and_y = (float *) _mm_malloc(block_size  * sizeof(float),VECALIGN);

    char mask_z_in_y_cutoff  = 0;
    char mask_z_in_x_cutoff = 0;

    float CYz_reduction, contains_tie, cutoff_distance;


    // initialize pointers for cache-block subcolumn vectors
    float *CXz;
    float *CYz;
    float *DXz;
    float *DYz;

    int y_block, x_block, offset;
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
            //  for(k = 0; k < block_size*block_size; ++k)
            //      conflict_block[k] = 0.;

            memops_loop_time += omp_get_wtime() - time_start;
            time_start = omp_get_wtime();
            DXz = D + x;
            DYz = D + y; // init pointers to subcolumns of D
            // __assume_aligned(conflict_block,64);
            // __assume_aligned(distance_block,64);
            // __assume(x_block % 16 == 0);
            for (z = 0; z < n; ++z) {
                // DXz = D + x + z*n;
                // DYz = D + y + z*n;
                // loop over all (i,j) pairs in block
                // _mm_prefetch(conflict_block + i + 32, 2);
                // _mm_prefetch(DXz + i + 32, 2);
                for (j = 0; j < y_block; ++j) {
                    ib = (x == y ? j : x_block);
                    // __assume_aligned(DXz, VECALIGN);
                    // __assume_aligned(distance_block, VECALIGN);
                    // __assume(x_block % 16 == 0);
                    for (i = 0; i < ib; ++i) {
                        // cutoff_distance = beta*distance_block[i + j * x_block];
                        // conflict_block[i + j * x_block] += ((DXz[i] <= cutoff_distance) | (DYz[j] <= cutoff_distance));
                        // mask_z_in_x_cutoff = (DXz[i] <= cutoff_distance);
                        // mask_z_in_y_cutoff = (DYz[j] <= cutoff_distance);
                        // //conflict_block[i + j * x_block] += (mask_z_in_y_cutoff | mask_z_in_x_cutoff);
                        // if (mask_z_in_x_cutoff || mask_z_in_y_cutoff){
                        //     ++conflict_block[i + j * x_block];
                        // }
                        if (DYz[j] <= beta * distance_block[i + j * x_block] || DXz[i] <= beta * distance_block[i + j * x_block]){
                            conflict_block[i + j * x_block] += 1.f;
                        //     //mask_z_in_conflict_focus[i] = 1.0f;
                        }
            
                    }
                    // __assume_aligned(conflict_block, VECALIGN);
                    // __assume(x_block % 16 == 0);
                    // for (i = 0; i < ib; ++i){
                    //     conflict_block[i + j* x_block] += mask_z_in_conflict_focus[i];
                    // }
                
                }

                // update pointers to subcolumns of D
                DXz += n;
                DYz += n;
            }
            for (k = 0; k < block_size*block_size; ++k)
                conflict_block[k] = 1.0f/conflict_block[k];
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
                    // z supports y+j
                    // __assume_aligned(distance_block,VECALIGN);
                    // __assume_aligned(DXz,VECALIGN);
                    // __assume(offset % 16 == 0);
                    // __assume(n % 16 == 0);
                    for (i = 0; i < ib; ++i) {
                        // mask_z_supports_x is when z support x.
                        mask_z_supports_x[i] = DXz[i] < DYz[j];
                        cutoff_distance = beta*distance_block[i + j * x_block];
                        mask_z_in_x_cutoff = (DXz[i] <= cutoff_distance);
                        mask_z_in_y_cutoff = (DYz[j] <= cutoff_distance);
                        mask_z_in_conflict_focus[i] = (mask_z_in_y_cutoff | mask_z_in_x_cutoff);
                    }
                    // __assume_aligned(mask_z_supports_x_and_y, VECALIGN);
                    // __assume(n % 16 == 0);
                    for (i = 0; i < ib; ++i) {
                    //     // mask_z_supports_x_and_y is when z supports both x and y. Support should be divided between x and y.
                        mask_z_supports_x_and_y[i] = DXz[i] == DYz[j] ? 1.0f : 0.0f;
                        contains_tie += mask_z_supports_x_and_y[i];
                    }
                    // // offset = j * x_block;
                    // for (i = 0; i < ib; ++i){
                        
                    // }

                    // for (i = 0; i<ib; ++i){
                    //     CXz[i] +=  conflict_block[i + j * block_size]*mask_z_in_conflict_focus[i]*(mask_z_supports_x[i]);
                    //     mask_z_supports_x[i] = 1 - mask_z_supports_x[i];
                    // }
                    
                    CYz_reduction = 0;
                    // __assume(offset % 16 == 0);
                    // __assume(n % 16 == 0);
                    // __assume_aligned(CXz,VECALIGN);
                    // __assume_aligned(CYz,VECALIGN);                    
                    // __assume_aligned(conflict_block,VECALIGN);                    
                    for (i =0; i<ib; ++i){
                        CXz[i] +=  conflict_block[i + j * x_block]*mask_z_in_conflict_focus[i]*(mask_z_supports_x[i]);
                        // 1 - mask_z_supports_x ==> z supports y and z supports both.
                        CYz_reduction +=  conflict_block[i + j * x_block]*mask_z_in_conflict_focus[i]*(1 - mask_z_supports_x[i]);
                    }
                    CYz[j] += CYz_reduction;

                    if (contains_tie > 0.5f){  
                        contains_tie = 0.f;                    
                        CYz_reduction = CYz[j];
                        for (i = 0; i < ib; ++i){
                            CXz[i] += 0.5f * conflict_block[i + j * x_block]*mask_z_in_conflict_focus[i]*mask_z_supports_x_and_y[i];
                            CYz_reduction -= 0.5f * conflict_block[i + j * x_block]*mask_z_in_conflict_focus[i]*mask_z_supports_x_and_y[i];
                            // mask_z_supports_x contains ties so subtract 1/2.
                        }
                        CYz[j] = CYz_reduction;
                    }
                    
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

    printf("==============================\n");
    printf("Sequential Loop Times\n");
    printf("==============================\n");

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


void pald_allz_openmp(float* restrict D, float beta, int n, float* restrict C, int block_size, int nthreads) {
    // declare indices
    // TODO: pragmas can't have curly braces in the same line!!!

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
            #pragma omp parallel for num_threads(nthreads) private(j, ib) schedule(monotonic:dynamic,8)
            for (j = 0; j < y_block; ++j) {
                // distance_block(:,j) = D(x:x+xb,y+j) in off-diagonal case
                ib = (x == y ? j : x_block); // handle diagonal blocks
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

            //#pragma omp parallel for num_threads(nthreads) private(k) schedule(static)
            for (k=0;k<block_size*block_size; ++k)
                conflict_block[k] = 1.0f/conflict_block[k];

            time_start = omp_get_wtime();
            #pragma omp parallel num_threads(nthreads) private(i, j, ib, z)
            {
                float *CXz, *CYz, *DXz, *DYz;
                float *mask_z_in_conflict_focus = (float *) _mm_malloc(block_size * sizeof(float),VECALIGN);
                float *mask_z_supports_x = (float *) _mm_malloc(block_size  * sizeof(float),VECALIGN);
                float *mask_z_supports_x_and_y = (float *) _mm_malloc(block_size  * sizeof(float),VECALIGN);

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
                        for (i = 0; i < ib; ++i) {
                            // mask_z_supports_x is when z support x.
                            mask_z_supports_x[i]= DXz[i] < DYz[j];
                        }
                        for (i = 0; i < ib; ++i) {
                            // mask_z_supports_x_and_y is when z supports both x and y. Support should be divided between x and y.
                            mask_z_supports_x_and_y[i] = DXz[i] == DYz[j] ? 1.0f:0.0f;
                            contains_tie += mask_z_supports_x_and_y[i];
                        }
                        // __assume(x_block % 16 == 0);
                        // __assume(n % 16 == 0);
                        for (i = 0; i < ib; ++i){
                            cutoff_distance = beta * distance_block[i + j * x_block];
                            mask_z_in_x_cutoff = (DXz[i] <= cutoff_distance);
                            mask_z_in_y_cutoff = (DYz[j] <= cutoff_distance);
                            mask_z_in_conflict_focus[i] = (mask_z_in_y_cutoff | mask_z_in_x_cutoff);
                        }
                        CXz = C + x + z*n;
                        CYz = C + y + z*n;
                        // __assume(x_block % 16 == 0);
                        // __assume(n % 16 == 0);
                        CYz_reduction = 0.f;
                        for (i = 0; i<ib; ++i){
                            CXz[i] += conflict_block[i + j * x_block]*mask_z_in_conflict_focus[i]*(mask_z_supports_x[i]);
                            // 1 - mask_z_supports_x ==> z supports y and z supports both.
                            CYz_reduction +=  conflict_block[i + j * x_block]*mask_z_in_conflict_focus[i]*(1 - mask_z_supports_x[i]);
                        }
                        CYz[j] += CYz_reduction;

                        if (contains_tie > 0.5f){  
                            contains_tie = 0.f;                    
                            CYz_reduction = CYz[j];
                            for (i = 0; i < ib; ++i){
                                CXz[i] += 0.5f * conflict_block[i + j * x_block]*mask_z_in_conflict_focus[i]*mask_z_supports_x_and_y[i];
                                // mask_z_supports_x contains ties so subtract 1/2.
                                CYz_reduction -= 0.5f * conflict_block[i + j * x_block]*mask_z_in_conflict_focus[i]*mask_z_supports_x_and_y[i];
                            }
                            CYz[j] = CYz_reduction;
                        }
                    }
                }
                _mm_free(mask_z_in_conflict_focus);
                _mm_free(mask_z_supports_x);
                _mm_free(mask_z_supports_x_and_y);
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
}

void pald_triplet_naive_openmp(float *D, float beta, int n, float *C, int num_threads){
    //TODO: Naive OpenMP triplet code.
}

void pald_triplet(float *D, float beta, int n, float *C, int block_size){
    //TODO: Optimized sequential triplet code.
}

void pald_triplet_openmp(float *D, float beta, int n, float *C, int block_size, int num_threads){
    //TODO: Optimized OpenMP triplet code.
}
