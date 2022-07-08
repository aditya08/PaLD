#!/bin/bash
export OMP_PROC_BIND=true
for matrix_size in 1024 2048 4096 8192
do
    for nthreads in 2 4 8 16 32
    do
	./PaLD_par_test $matrix_size 256 256 $nthreads 2>&1 | tee strong_scaling.txt
    sleep 2
    done
done
