#!/bin/bash

# This script is used for listing each archives and
# save their filelist to file.

SUPPROTED_FORMAT=(7z)
SUPPROTED_FORMAT+=(gz gzip tgz)
SUPPROTED_FORMAT+=(bz2 bzip2 tbz2 tbz)
SUPPROTED_FORMAT+=(tar vhd img)
SUPPROTED_FORMAT+=(rar)
SUPPROTED_FORMAT+=(xz txz)

params=" -iname *.zip"
printf "Supported format: zip"
for fmt in ${SUPPROTED_FORMAT[@]}
do
  params+=" -o -iname *.$fmt "
  printf " $fmt"
done
#echo $params
echo

if [ -d "$1" ]
then
    find "$1" -type f $params | \
    while read fullpath
    do
    filename="$(basename "$fullpath")"
    echo "$filename"  "$fullpath"
    # probably bug for long filename here 
    # bug: cannot skip password protected filelist
    7z l -r "$fullpath" > "/tmp/$filename.filelist" && \
        mv --backup=numbered "/tmp/$filename.filelist" "$fullpath.filelist"
    done
elif [ -f "$1" ]
then
  filename="$(basename "$1")"
    echo "$filename"  "$1"
    # probably bug for long filename here 
    # bug: cannot skip password protected filelist
    7z l -r "$1" > "/tmp/$filename.filelist" && \
        mv --backup=numbered "/tmp/$filename.filelist" "$1.filelist"
else
  echo "Param($1) not a file or dir"
fi
