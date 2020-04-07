#!/bin/bash

# All maps
declare -a maps=("Black Woods" "Cliffside Bay" "Diamond Coast" "Foggy Hills" "Grand River" "Green Plains" "Islands" "Lagoon Shore" "Riverrun" "Sandy Beach" "Shady Strands" "Two Rivers")

count="${#maps[@]}"

# Print 5 random maps
for i in 1 2 3 4 5
do
  cur="$(( $RANDOM % $count ))"
  echo "[$((cur+1))] ${maps[$cur]}"
done
