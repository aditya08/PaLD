#!/bin/bash

for matrix_size in 500 1000 1500 2000 2500 3000 3500 4000 4500 5000
do
    for cache_size in 256
    do
	./PaLD_test $matrix_size $cache_size >> new.txt
    done
done
