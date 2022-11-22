#!/bin/bash
export OMP_PROC_BIND=true
matrix_size=$1
for nthreads in 2 4 8 16 32
do
./PaLD_par_test $matrix_size 256 256 $nthreads 5 2>&1 | tee -a output.txt
sleep 5
done