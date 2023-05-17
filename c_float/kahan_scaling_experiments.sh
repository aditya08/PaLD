#!/bin/bash
./allz_strong_scaling.sh 512
sleep 2
./allz_weak_scaling.sh
sleep 2
./triplet_strong_scaling.sh 512
sleep 2
./triplet_weak_scaling.sh
