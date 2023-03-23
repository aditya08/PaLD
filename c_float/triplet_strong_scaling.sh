#!/bin/bash
matrix_size=$1
omp_block_size=$2
while [ ${matrix_size} -lt 8193 ]
do
    #./PaLD_triplet_benchmark $matrix_size 256 5 2>&1 | tee -a ./logs/triplet_strong_scaling/sequential/triplet_matsize_${matrix_size}_strong_scaling.txt

    while [ ${omp_block_size} -lt 257 ]
    do
        export OMP_PROC_BIND=true
        for nthreads in 2 4 8 16 32 48
        do
            #echo "${matrix_size} ${omp_block_size} ${nthreads}"
            ./PaLD_triplet_omp_benchmark $matrix_size $omp_block_size $nthreads 5 2>&1 | tee -a ./logs/triplet_strong_scaling/openmp/triplet_matsize_${matrix_size}_ompblk_${omp_block_size}_threads_${nthreads}_strong_scaling.txt
            sleep 5
        done
        omp_block_size=`expr $omp_block_size + 32`
    done
    matrix_size=`expr $matrix_size + 512`
    omp_block_size=$2
done
