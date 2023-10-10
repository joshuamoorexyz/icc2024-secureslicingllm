#!/bin/bash

# Function to run iperf3 with the specified bandwidth
run_iperf() {
    sudo ip netns exec ue1 iperf3 -c 172.16.0.1 -p 5401 -i 1 -t 10 -R -b "$1"
}

# Infinite loop to switch between high and low bandwidth every 10 seconds
while true; do
    # Run iperf3 with high bandwidth (30M)
    run_iperf "30M"
    
    # Sleep for 10 seconds
    #sleep 10
    
    # Run iperf3 with low bandwidth (e.g., 1M, adjust as needed)
    run_iperf "1M"
    
    # Sleep for 10 seconds
    sleep 10
done
