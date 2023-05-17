#!/bin/bash
matrix_size=$1
omp_block_size=256
# sequential and openmp runs (openmp runs without numa-awareness)
unset KMP_AFFINITY
while [ ${matrix_size} -lt 8193 ]
do
    echo "${matrix_size}"
    ./PaLD_allz_benchmark $matrix_size 256 5 2>&1 | tee -a ./logs/kahan/allz_strong_scaling/sequential/allz_matsize_${matrix_size}_blk_256_sequential_strong_scaling.txt

    while [ ${omp_block_size} -lt 257 ]
    do
        for nthreads in 2 4 8 16 32 #48
        do
            # echo "${matrix_size} ${omp_block_size} ${nthreads} ${KMP_AFFINITY}"
            ./PaLD_allz_omp_benchmark $matrix_size $omp_block_size $nthreads 5 2>&1 | tee -a ./logs/kahan/allz_strong_scaling/openmp/allz_matsize_${matrix_size}_blk_${omp_block_size}_threads_${nthreads}_strong_scaling.txt
            sleep 2
        done
        omp_block_size=`expr $omp_block_size + 32`
    done
    matrix_size=`expr ${matrix_size} \\* 2`
    omp_block_size=256
done

matrix_size=$1
# numa-aware openmp runs (using KMP_AFFINITY env. var)
export KMP_AFFINITY="granularity=fine,compact,1,0"
while [ ${matrix_size} -lt 8193 ]
do
    # ./PaLD_allz_benchmark $matrix_size 256 5 2>&1 | tee -a ./logs/allz_strong_scaling/sequential/allz_matsize_${matrix_size}_strong_scaling.txt
    while [ ${omp_block_size} -lt 257 ]
    do
        for nthreads in 2 4 8 16 32 #48
        do
            # echo "${matrix_size} ${omp_block_size} ${nthreads} ${KMP_AFFINITY}"
            ./PaLD_allz_omp_benchmark $matrix_size $omp_block_size $nthreads 5 2>&1 | tee -a ./logs/kahan/allz_strong_scaling/numa_openmp/allz_numa_matsize_${matrix_size}_blk_${omp_block_size}_threads_${nthreads}_strong_scaling.txt
            sleep 2
        done
        omp_block_size=`expr $omp_block_size + 32`
    done
    matrix_size=`expr $matrix_size \\* 2`
    omp_block_size=256
done
unset KMP_AFFINITY