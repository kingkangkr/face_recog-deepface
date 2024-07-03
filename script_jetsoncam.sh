#!/bin/bash

# Set IP address
sudo ifconfig eth0 192.168.1.1 netmask 255.255.255.0 up
if [ $? -eq 0 ]; then
    echo "IP address assigned successfully."
else
    echo "Failed to assign IP address."
    exit 1
fi

# Disable Network Manager for eth0
sudo nmcli dev set eth0 managed no
if [ $? -eq 0 ]; then
    echo "Network Manager disabled for eth0 successfully."
else
    echo "Failed to disable Network Manager for eth0."
    exit 1
fi
# Ping the specified IP
# lan cable = 192.168.1.2
# Check the ping result
if [ $? -eq 0 ]; then
    echo "Communication successful, starting server_face_recognition.py"
    python3 server_depface_jetsoncam.py
else
    echo "Communication failed"
fi
