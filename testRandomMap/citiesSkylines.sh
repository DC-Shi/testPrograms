#!/bin/bash

# All base maps
declare -a maps


for mapfile in CSL*.mapnames
do
  curPrefix="${mapfile%%.*}"
  while read line;
  do
    
    maps+=("$curPrefix $line")
  done <$mapfile
done 
count="${#maps[@]}"

# Print 5 random maps
for i in 1 2 3 4 5
do
  cur="$(( $RANDOM % $count ))"
  echo "${maps[$cur]}"
done
