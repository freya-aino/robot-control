#!/usr/bin/bash

if [[ -z "$1" ]]; then
    echo "❌ No argument supplied."
else
    echo "✅ trying to add 192.168.123.201/24 to device: $1"
    sudo ip a add 192.168.123.201/24 dev $1
fi
