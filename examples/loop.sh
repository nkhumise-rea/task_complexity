#!/bin/bash

count=0
max_count=56
while [ $count -le $max_count ]
do  
    echo $count
    # python random_sampling_HER.py --env two_link  --n_units 32 --count $count --n_samples 100 --n_episodes 500
    python random_sampling_HER.py --env one_link --n_units 32 --count $count --n_samples 100 --n_episodes 5
    count=$((count + 1))
    #--multiprocess 8
done
