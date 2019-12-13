#!/bin/sh

echo "===== Building MoMC     ====="
gcc -O3 -DMOMC -o MoMC MoMC2016.c

echo "===== Building akmaxsat ====="
cd akmaxsat_1.1
make cleanup
make
mv akmaxsat ../
cd ../
