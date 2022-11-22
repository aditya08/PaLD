#!/bin/bash
export OMP_PROC_BIND=true
matrix_size=$1
nthreads=$2

./PaLD_par_test $matrix_size 256 256 $nthreads 5 2>&1 | tee -a allz_weak_scaling.txt
