#!/bin/bash

while read -r line
do
  gflops=$(echo $line | grep -o [0-9.]*e+03)
  tflops=$(echo $gflops | sed -e 's/e+03$//')
#${gflops%"e+03"}
#  echo "tflops=$tflops"
  if [[ ! -z "$tflops" ]] && (( $(echo "$tflops > 6.3" | bc -l) ))
  then
    echo $line
  fi
done < "result.table"

