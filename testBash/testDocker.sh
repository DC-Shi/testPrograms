#!/bin/bash
# This script is to debug why no GPU after exporting.


pause()
{
  read  -r -p "Press Enter key to continue..." key
}


highlight()
{
  echo -e "\e[4;40;34m $1 \e[0m"
}

# http://tldp.org/LDP/abs/html/debugging.html#ASSERT
#######################################################################
assert ()                 #  If condition false,
{                         #+ exit from script
                          #+ with appropriate error message.
  E_PARAM_ERR=98
  E_ASSERT_FAILED=99


  if [ -z "$2" ]          #  Not enough parameters passed
  then                    #+ to assert() function.
    return $E_PARAM_ERR   #  No damage done.
  fi

  lineno=$2

  $1

  if [[ ! $? ]] 
  then
    echo "Assertion failed:  \"$1\""
    echo "File \"$0\", line $lineno"    # Give name of file and line number.
    exit $E_ASSERT_FAILED
  else
    highlight "$1"          # print current command line
    return
  #   and continue executing the script.
  fi  
} # Insert a similar assert() function into a script you need to debug.    
#######################################################################

highlight 'Please run nvidia-smi in docker, you should see GPU'
#echo nvidia-docker run --name test_aes_nvi -p 33000:3300 -p 2200:22 -d -i nvidia/cuda:8.0-devel /bin/bash

#echo nvidia-docker exec -it test_aes_nvi /bin/bash

assert "echo 4 -lt 3" $LINENO

assert "nvidia-docker run -it --rm --name test_aes_nvi -p 33000:3300 -p 2200:22  -i nvidia/cuda:8.0-devel nvidia-smi" $LINENO

highlight "export image to tar ball file"
nvidia-docker export test_aes_nvi > backup.tar

highlight "re-import image"
assert "docker rmi nvidia_aes:aes" $LINENO
cat backup.tar | docker import - nvidia_aes:aes

assert "nvidia-docker run -it --rm --name restored_aes_nvi -p 33000:3300 -p 2200:22  -i nvidia_aes:aes nvidia-smi" $LINENO
#assert "nvidia-docker run --name restored_aes -p 33001:3300 -p 2201:22 -d -i nvidia_aes:aes /bin/bash" $LINENO
#assert "nvidia-docker exec -it restored_aes nvidia-smi" $LINENO
