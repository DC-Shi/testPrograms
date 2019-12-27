#!/bin/bash

# Remove exit command first
rm -f cmd/exit

# Deal with space file name
OIFS="$IFS"
IFS=$'\n'

while true
do
  for file in $(find cmd/ -type f)
  do
    IFS=$OIFS
    exe=$(basename -- "$file")

    $exe
    result=$?
    IFS=$'\n'
    if [ $result -eq 0 ]; then
      echo "Found $exe, =========Success"
    else
      echo "Found $exe, *********Failed with $result"
    fi
    rm "$file"
  done
  sleep 1
done

IFS="$OIFS"
