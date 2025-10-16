#!/bin/bash

count=0
max_count=50
while [ $count -le $max_count ]
do  
    echo $count
    # python rwg_1.py --count $count
    python rwg_1T55.py --count $count
    # python rwg_HER_1.py --count $count
    count=$((count + 1))
    #--multiprocess 8
done
