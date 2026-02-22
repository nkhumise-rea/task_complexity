#!/bin/bash

count=46
max_count=0
while [ $count -ge $max_count ]
do  
    echo $count
    # python rwg.py --count $count
    # python rwg_HER.py --count $count
    python rwg_obstacle.py --count $count
    count=$((count - 1))
    #--multiprocess 8
done
