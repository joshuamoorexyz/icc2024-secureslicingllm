import numpy as np
import matplotlib.pyplot as plt

# Function to extract bandwidth data from a text file
def extract_bandwidth(filename):
    bandwidth = []

    with open(filename, 'r') as file:
        lines = file.readlines()

        for line in lines:
            # Check if the line contains 'sec' and 'Mbits/sec'
            if ' sec ' in line and ' Mbits/sec ' in line:
                parts = line.split()
                bitrate_index = parts.index('Mbits/sec')
                bandwidth_value = float(parts[bitrate_index - 1])
                bandwidth.append(bandwidth_value)

            if ' sec ' in line and ' Kbits/sec ' in line:
                parts = line.split()
                bitrate_index = parts.index('Kbits/sec')
                bandwidth_value = float(parts[bitrate_index - 1])
                bandwidth_value = bandwidth_value / 1024
                bandwidth.append(bandwidth_value)


    return bandwidth

# Extract bandwidth data from the text files
bandwidth1 = extract_bandwidth('UE3-maliciousdefault')
#bandwidth2 = extract_bandwidth('ue2-default-throughput')

# Plot the data
plt.plot(bandwidth1, label='Bandwidth1', linewidth=1)
#plt.plot(bandwidth2, label='Bandwidth2', linewidth=1)

plt.xlabel('Measurement Interval')
plt.ylabel('Bandwidth (Mbits/sec)')
plt.legend()
plt.show()
