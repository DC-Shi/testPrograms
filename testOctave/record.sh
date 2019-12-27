#!/bin/bash

summary_train () {
  FILENAME='train.txt'
  
  if [[ "$1" ]]
  then
    FILENAME="$1"
  fi
  
  totalsec="0"
  
  echo "========================= $FILENAME ================="

  # in a subshell, this will address \r problems  
  cat "$FILENAME" | while read -r line
  do
    # Get last result if it was updated multiple times
    output=$(echo "$line"|awk -F'\r' '{print $NF}')
    time_str=$(echo "$output"|grep -o "[0-9]*m [0-9]*")
  
    if [[ ! -z "$time_str" ]]
    then
      minute="${time_str%m*}"
      seconds="${time_str##*m}"
      convertsec="$((60*minute + seconds))"
      totalsec="$((totalsec + convertsec))"
      echo "$output = $convertsec seconds, total = $totalsec"
    fi
  done
}
summary_dmon () {
  FILENAME='dmon.txt'
  
  if [[ "$1" ]]
  then
    FILENAME="$1"
  fi
  
  echo "========================= $FILENAME ================="

  # in a subshell, this will address \r problems  
  grep "^[^#]" "$FILENAME" | awk '{ pwr += $2; gtemp += $3; pclk += $10; count++ } END { print "avg_pwr = ", pwr/count, "avg_gtemp = ", gtemp/count, "avg_pclk = ", pclk/count }'
}


BATCH_PER_GPU=64

# Record one training and monitoring
# param1: gpu list to use
# param2: fp16 enabled or not
# param3: GPU frequency in MHz
record_one () {
  # GPU list
  gpu_ids="$1"
  totalgpus=$(( $(echo "$gpu_ids"|grep -o "," |wc -l) + 1 ))
  # FP16
  fp16=$2
  fp16str=$([ "$fp16" == 1 ] && echo "FP16" || echo "FP32")
  # GPU frequency
  if [[ "$3" ]]
  then
    gpu_freq="$3"
    echo nvidia-smi -lgc $gpu_freq -i $gpu_ids
    nvidia-smi -lgc $gpu_freq -i $gpu_ids
  fi
  # log file names
  filename="${totalgpus}GPU_${fp16str}_${BATCH_PER_GPU}B_${gpu_freq}MHz"
  dmonlog="results/${filename}_dmon.log"
  trainlog="results/${filename}_train.log"
  
  # shift two params
  shift;shift
  # Start monitoring process
  nvidia-smi dmon -i ${gpu_ids} -f $dmonlog &
  mon_pid=$!
  
  # Start training process
  # timing
  time_start=$(date +%s)
  echo "running: python3 train.py  --batchsize $((BATCH_PER_GPU*totalgpus)) --gpu_ids ${gpu_ids} --fp16 ${fp16} --num_epochs 5 "
  python3 train.py  --batchsize $((BATCH_PER_GPU*totalgpus)) --gpu_ids ${gpu_ids} --fp16 ${fp16} --num_epochs 5 >$trainlog &
  train_pid=$!
  
  echo waiting $train_pid, and $mon_pid in background
  wait $train_pid

  kill $mon_pid
  time_end=$(date +%s)
  echo $((time_end - time_start)) >> $trainlog
  echo "Flushing buffers..."
  # do sync before reading file, or files could be empty
#  sync

#  echo running summary on $trainlog
#  summary_train "$trainlog"
#  summary_dmon "$dmonlog"
  

}


# Summary with formatted
# param1: filename(should be end with dmon.log)
SummaryFormatted () {
  dmonlog="$1"
  # Get counts
  gpucnt=$(echo "$dmonlog" | grep -o "[0-9]*GPU_" | sed -e 's/GPU_//g' )
  fp16=$(echo "$dmonlog" | grep -o "_FP[0-9]*" | sed -e 's/_//g' )
  # Get training logfile
  trainlog="${dmonlog/dmon.log/train.log}"


  if [[ -f "$trainlog" ]]
  then
    epoch="-1"
    epoch_time=0
    totalsec=0
    while read -r line
    do
      # Get last result if it was updated multiple times
      output=$(echo "$line"|awk -F'\r' '{print $NF}')
      time_str=$(echo "$output"|grep -o "[0-9]*m [0-9]*")
     
      if [[ ! -z "$time_str" ]]
      then
        minute="${time_str%m*}"
        seconds="${time_str##*m}"
        convertsec="$((60*minute + seconds))"
        totalsec="$((totalsec + convertsec))"
        #echo "$output = $convertsec seconds, total = $totalsec"
        if [[ $epoch -ge 0 ]]
        then
          epoch_time=$(( epoch_time + convertsec))
        fi
        epoch=$(( epoch + 1 ))
      fi
    done <"$trainlog"

    # Assume last line of log is total time in seconds.
    finaltime=$(tail -n1 $trainlog)
  else
    echo "No training log: $trainlog"
  fi

  #echo "trainlog = $trainlog"
  #echo "gpucnt = $gpucnt"
  #echo "fp16 = $fp16"
  #echo "epoch = $epoch"
  #echo "epoch_time = $epoch_time"
  #echo "totalsec = $totalsec"

  if [[ -f "$dmonlog" ]]
  then
    # Print the power related
    grep "^[^#]" "$dmonlog" | awk \
    -v gpucnt="$gpucnt" -v fp16="$fp16" -v finaltime="$finaltime" -v epoch="$epoch" -v epoch_time="$epoch_time" -v totalsec="$totalsec" \
    '{ 
      pwr += $2;
      gtemp += $3;
      pclk += $10;
      count++;
    } END {
        printf ("%4s\t%4s\t%4s\t%10s\t%9s\t%7s\t%8s\t%8s\n", gpucnt, fp16, -1, totalsec, epoch_time/epoch, pwr/count, gtemp/count, pclk/count)
    }'
  else
    echo "No dmon log: $dmonlog"
  fi
}

if [[ "$SUMMARYONLY" ]]
then
  echo "card	prec	time	epoch_time	epoch_avg	avg_pwr	avg_gtemp	avg_pclk" 1>&2;
  for dmonlog in results/*_dmon.log
  do
    SummaryFormatted "$dmonlog"
  done

  exit 0
fi



for gpu in "0,3" "0"
do
  for fp16 in 1 0
  do
    for gpufreq in 1530 1470 1380 1290 1200 1110 1020 960
    do
      record_one "$gpu" "$fp16" "$gpufreq"
      # The last file is zero, this might because file is not flushed before container shutdown.
      sleep 10
      sync
    done
  done
done
