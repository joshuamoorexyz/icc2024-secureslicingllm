#!/bin/bash

# Function to run iperf3 with the specified bandwidth
run_iperf() {
    sudo ip netns exec ue1 iperf3 -c 172.16.0.1 -p 5006 -i "$1" -t 10 -R -b 30
}

# Infinite loop to switch between high and low bandwidth every 10 seconds
while true; do
    # Run iperf3 with low latency
    run_iperf "0.1"
    
    # Sleep for 10 seconds
    #sleep 10
    
    # Run iperf3 with high latency
    run_iperf "1"
    
    # Sleep for 10 seconds
    sleep 10
done
