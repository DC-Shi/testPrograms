#!/usr/bin/env bash

good=0
bad=0
total=0

while true; do
{
  ip="192.168.415.19"
  ping -c1 -W1 $ip 2>/dev/null 1>/dev/null
  ans=$?
  total=$((total + 1))
  
  if [[ $ans -eq 0 ]]; then # Good!
    good=$((good + 1))
    percent=$(echo "scale=2; 100* ${good}/${total}" |bc)
    printf "\rGood (${good}/${total}=${percent}%%) Last update: %s  " "$(date)"
  else # Failed!
    bad=$((bad + 1))
    echo "Bad  (${bad}/${total}) Failed to ping on $(date)"
  fi

  # Check whether this IP has been in known_hosts
  ssh-keygen -F ${ip}
  # Get the public key
  ssh-keygen -T 5 ${ip}

  sleep 1
}
done
