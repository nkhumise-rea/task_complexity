#!/bin/bash

count=0
max_count=50
while [ $count -le $max_count ]
do  
    echo $count
    # python rwg.py --count $count
    python rwg_HER.py --count $count
    count=$((count + 1))
    #--multiprocess 8
done
