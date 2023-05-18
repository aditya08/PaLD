#!/bin/bash
matrix_size=$1

# sequential and openmp runs (openmp runs without numa-awareness)
unset KMP_AFFINITY
while [ ${matrix_size} -lt 8193 ]
do
    # echo "${matrix_size}"
    ./PaLD_triplet_benchmark $matrix_size 256 128 1 2>&1 | tee ./logs/kahan/triplet_strong_scaling/sequential/triplet_matsize_${matrix_size}_ublk_256_cohblk_128_sequential_strong_scaling.txt

    for conflict_block_size in 64 96 128 192 256
    do
        for cohesion_block_size in 64 96 128 192 256
        do
        ## TODO: create nested loop to test different conflict and cohesion block sizes: 64 96 128 192 256
            for nthreads in 2 4 8 16 32 #48
            do
                # echo "${matrix_size} ${conflict_block_size} ${cohesion_block_size} ${nthreads} ${KMP_AFFINITY}"
                ./PaLD_triplet_omp_benchmark $matrix_size $conflict_block_size $cohesion_block_size $nthreads 1 2>&1 | tee ./logs/kahan/triplet_strong_scaling/openmp/triplet_matsize_${matrix_size}_ublk_${conflict_block_size}_cohblk_${cohesion_block_size}_threads_${nthreads}_strong_scaling.txt
                sleep 2
            done
        done
    done
    matrix_size=`expr ${matrix_size} \\* 2`
done

matrix_size=$1
# numa-aware openmp runs (using KMP_AFFINITY env. var)
export KMP_AFFINITY="granularity=fine,compact,1,0"
while [ ${matrix_size} -lt 8193 ]
do
    # ./PaLD_triplet_benchmark $matrix_size 256 5 2>&1 | tee -a ./logs/triplet_strong_scaling/sequential/triplet_matsize_${matrix_size}_strong_scaling.txt
    ## TODO: create nested loop to test different conflict and cohesion block sizes: 64 96 128 192 256
    for conflict_block_size in 64 96 128 192 256
    do
        for cohesion_block_size in 64 96 128 192 256
        do

            for nthreads in 2 4 8 16 32 #48
            do
                # echo "${matrix_size} ${conflict_block_size} ${cohesion_block_size} ${nthreads} ${KMP_AFFINITY}"
                ./PaLD_triplet_omp_benchmark $matrix_size $conflict_block_size $cohesion_block_size $nthreads 1 2>&1 | tee ./logs/kahan/triplet_strong_scaling/numa_openmp/triplet_numa_matsize_${matrix_size}_ublk_${conflict_block_size}_cohblk_${cohesion_block_size}_threads_${nthreads}_strong_scaling.txt
                sleep 1
            done
        done
    done
    matrix_size=`expr $matrix_size \\* 2`
done
unset KMP_AFFINITY