#!/usr/bin/env bash

about()
{
	echo $0 [port] [response]
	echo Run simple HTTP server to show other server status.
	echo Optional arguments:
	echo "[port]     default to 41519"
	echo "[response] default to Server ready"
}


about
echo "Press Ctrl-C to stop server."

while true; do
{
	# Print first part
	echo -e 'HTTP/1.1 200 OK\r\n'
        echo "<meta http-equiv=\"content-type\" content=\"text/html; charset=UTF-8\">" # For unicode compatibality
       	echo "<style> th, td {border-bottom: 1px solid black;}</style>" # Add style for table
        echo "<h1> ${2:-Server Ready} </h1>" # Head
	# Scan each host and decide status
	# Create table
	echo "<table>"
	# table header
	echo "<tr><th>Server</th><th>Status</th></tr>"
	while read ip
	do
		# table row
		echo "<tr>"
		echo "<p hidden>"
		ping -c1 -W1 $ip 2>&1
		ans=$?
		echo "</p>"
		if [[ $ans -eq 0 ]]; then
			status='<font color="green">GOOD!</font>'
		else
			status='<font color="red">offline</font>'
		fi
		echo "<td>$ip</td><td>$status</td>"
		echo "</tr>"
	done <serverlist
	echo "</table>"
	# Finish the page with timestamp
	echo "<p> Updated on: $(date)"
} | nc -l "${1:-41519}" -q 1  # -l for port, -q for timeout
done
