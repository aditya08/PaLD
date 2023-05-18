#!/bin/bash
unset KMP_AFFINITY
# sequential and openmp allz weak scaling runs (without numa-awareness)
for matrix_size in 256 512 1024
do
    ./PaLD_triplet_benchmark $matrix_size 256 128 1 2>&1 | tee ./logs/kahan/triplet_weak_scaling/sequential/triplet_matsize_${matrix_size}_ublk_256_cohblk_128_sequential_weak_scaling.txt
    # echo "${matrix_size}"

    for conflict_block_size in 64 96 128 192 256
    do
        for cohesion_block_size in 64 96 128 192 256
        do
            for nthreads in 2 4 8 16 32
            do
                work=$(echo ${matrix_size} ${nthreads} | awk '{ print $1^(3)*$2 }')
                omp_matrix_size=$(echo ${work} | awk '{ print int($1^(1/3) + 0.5) }')
                ./PaLD_triplet_omp_benchmark $omp_matrix_size $conflict_block_size $cohesion_block_size $nthreads 1 2>&1 | tee ./logs/kahan/triplet_weak_scaling/openmp/triplet_matsize_${omp_matrix_size}__ublk_${conflict_block_size}_cohblk_${cohesion_block_size}_threads_${nthreads}_weak_scaling.txt
                #echo "${nthreads} ${omp_matrix_size} ${conflict_block_size} ${cohesion_block_size}"
                sleep 2
            done
        done
    done
done
    #matrix_size=`expr $matrix_size`

export KMP_AFFINITY="granularity=fine,compact,1,0"
# openmp allz weak scaling with numa-awareness runs (using KMP_AFFINITY env. var.)
for matrix_size in 256 512 1024
do
    ## TODO: create nested loop to test different conflict and cohesion block sizes: 64 96 128 192 256
    for conflict_block_size in 64 96 128 192 256
    do
        for cohesion_block_size in 64 96 128 192 256
        do
            for nthreads in 2 4 8 16 32
            do
                work=$(echo ${matrix_size} ${nthreads} | awk '{ print $1^(3)*$2 }')
                omp_matrix_size=$(echo ${work} | awk '{ print int($1^(1/3) + 0.5) }')
                ./PaLD_triplet_omp_benchmark $omp_matrix_size $conflict_block_size $cohesion_block_size $nthreads 1 2>&1 | tee ./logs/kahan/triplet_weak_scaling/numa_openmp/triplet_matsize_${omp_matrix_size}_ublk_${conflict_block_size}_cohblk_${cohesion_block_size}_threads_${nthreads}_weak_scaling.txt
                #echo "${nthreads} ${omp_matrix_size} ${conflict_block_size} ${cohesion_block_size}"
                sleep 2
            done
        done
    done
    #matrix_size=`expr $matrix_size`
done
unset KMP_AFFINITY