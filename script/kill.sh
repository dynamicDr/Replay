#!/bin/bash
pid=$(ps aux | grep "challenge.py" | grep -v grep | awk '{print $2}')
if [ ! -z "$pid" ]; then
    echo "Killing process $pid"
    kill -9 $pid
else
    echo "No process found"
fi

pid=$(ps aux | grep "train.py" | grep -v grep | awk '{print $2}')
if [ ! -z "$pid" ]; then
    echo "Killing process $pid"
    kill -9 $pid
else
    echo "No process found"
fi
