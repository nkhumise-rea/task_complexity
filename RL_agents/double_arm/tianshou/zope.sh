#!/bin/bash

count=0
max_count=5
while [ $count -le $max_count ]
do  
    echo $count
    # python sac_test.py --count $count
    # python sac_test_HER.py --count $count
    python sac_test_obstacle.py --count $count
    count=$((count + 1))
    #--multiprocess 8
done