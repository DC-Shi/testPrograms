#!/bin/bash

CUFILE="kernel.cu"
NUMVAR=1023

# Clear file content
echo "" > $CUFILE

# Output kernel header
cat >> $CUFILE <<EOT
#include "kernel.cuh"

__global__ void kernel_64reg(double * writeOut)
{
    // 64K*32bit for 1 block, 64K*32bit/64bit/1024t = 32 double
    // 255*32bit max reg size for single thread.
EOT

# Kernel environment variable declariation
for ((varIdx=0; varIdx<=$NUMVAR; varIdx++))
do
  echo "volatile double r$(printf %03x $varIdx) = $varIdx;" >> $CUFILE
done

# add the number, even add to odd

for ((varIdx=0; varIdx<=$NUMVAR; varIdx=varIdx+2))
do
  echo "r$(printf %03x $varIdx) = r$(printf %03x $((varIdx+1))) + r$(printf %03x $varIdx);" >> $CUFILE
done

# Write result to buffer
for ((varIdx=0; varIdx<=$NUMVAR; varIdx++))
do
  echo "writeOut[$varIdx] = r$(printf %03x $varIdx);" >> $CUFILE
done

# Output kernel ending
cat >> $CUFILE <<EOT

}

EOT