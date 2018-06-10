#!/bin/bash
echo "===================CIL Text Classification====================="
echo "Warning: if you have not connected to the ETH VPN, please do so before running this script!"
echo "===================Downloading dataset========================="

# Move to where this script is located
cd "$(dirname $0)"

wget -N "http://www.da.inf.ethz.ch/teaching/2018/CIL/material/exercise/twitter-datasets.zip"
unzip -o twitter-datasets.zip

cd -
