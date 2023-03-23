#!/bin/bash
omp_block_size=$1
for matrix_size in 256
do
    #./PaLD_triplet_benchmark $matrix_size 256 5 2>&1 | tee -a ./logs/triplet_weak_scaling/${matrix_size}/triplet_matsize_${matrix_size}_sequential_weak_scaling.txt
    #echo "${matrix_size}"

    while [ ${omp_block_size} -lt 257 ]
    do
        export OMP_PROC_BIND=true
        for nthreads in 24
        do
            work=$(echo ${matrix_size} ${nthreads} | awk '{ print $1^(3)*$2 }')
            omp_matrix_size=$(echo ${work} | awk '{ print int($1^(1/3) + 0.5) }')
            ./PaLD_triplet_omp_benchmark $omp_matrix_size $omp_block_size $nthreads 5 2>&1 | tee -a ./logs/triplet_weak_scaling/${matrix_size}/triplet_matsize_${omp_matrix_size}_ompblk_${omp_block_size}_threads_${nthreads}_weak_scaling.txt
            #echo "${nthreads} ${omp_matrix_size} ${omp_block_size}"
            sleep 5
        done
        omp_block_size=`expr $omp_block_size + 32`
    done
    #matrix_size=`expr $matrix_size`
    omp_block_size=$1
done