#!/bin/bash

# Set IP address
sudo ifconfig eno1 192.168.1.2 netmask 255.255.255.0 up
if [ $? -eq 0 ]; then
    echo "IP address assigned successfully."
else
    echo "Failed to assign IP address."
    exit 1
fi

# Disable Network Manager for eth0
sudo nmcli dev set eno1 managed no
if [ $? -eq 0 ]; then
    echo "Network Manager disabled for eth0 successfully."
else
    echo "Failed to disable Network Manager for eth0."
    exit 1
fi

# Execute client_camera.py
python3 client_jetsoncam.py
if [ $? -eq 0 ]; then
    echo "client_camera.py executed successfully."
else
    echo "Failed to execute client_camera.py."
fi
