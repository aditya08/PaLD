#!/bin/bash
omp_block_size=256
unset KMP_AFFINITY
# sequential and openmp allz weak scaling runs (without numa-awareness)
for matrix_size in 256 512 1024
do
    # mkdir -p ./logs/kahan/allz_weak_scaling/${matrix_size}/
    ./PaLD_allz_benchmark $matrix_size 256 1 2>&1 | tee ./logs/kahan/allz_weak_scaling/sequential/allz_matsize_${matrix_size}_blk_256_sequential_weak_scaling.txt
    echo "${matrix_size}"

    while [ ${omp_block_size} -lt 257 ]
    do
        for nthreads in 2 4 8 16 32
        do
            work=$(echo ${matrix_size} ${nthreads} | awk '{ print $1^(3)*$2 }')
            omp_matrix_size=$(echo ${work} | awk '{ print int($1^(1/3) + 0.5) }')
            ./PaLD_allz_omp_benchmark $omp_matrix_size $omp_block_size $nthreads 1 2>&1 | tee ./logs/kahan/allz_weak_scaling/openmp/allz_matsize_${omp_matrix_size}_ompblk_${omp_block_size}_threads_${nthreads}_weak_scaling.txt
            echo "${nthreads} ${omp_matrix_size} ${omp_block_size} ${KMP_AFFINITY}"
            sleep 2
        done
        omp_block_size=`expr $omp_block_size + 32`
    done
    #matrix_size=`expr $matrix_size`
    omp_block_size=256
done

export KMP_AFFINITY="granularity=fine,compact,1,0"
# openmp allz weak scaling run with numa-awareness (using KMP_AFFINITY env var.)
for matrix_size in 256 512 1024
do
    while [ ${omp_block_size} -lt 257 ]
    do
        for nthreads in 2 4 8 16 32
        do
            work=$(echo ${matrix_size} ${nthreads} | awk '{ print $1^(3)*$2 }')
            omp_matrix_size=$(echo ${work} | awk '{ print int($1^(1/3) + 0.5) }')
            ./PaLD_allz_omp_benchmark $omp_matrix_size $omp_block_size $nthreads 1 2>&1 | tee ./logs/kahan/allz_weak_scaling/numa_openmp/allz_matsize_${omp_matrix_size}_ompblk_${omp_block_size}_threads_${nthreads}_weak_scaling.txt
            echo "${nthreads} ${omp_matrix_size} ${omp_block_size} ${KMP_AFFINITY}"
            sleep 2
        done
        omp_block_size=`expr $omp_block_size + 32`
    done
    #matrix_size=`expr $matrix_size`
    omp_block_size=256
done

unset KMP_AFFINITY