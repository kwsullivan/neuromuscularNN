#!/bin/bash

CONFIG_PATH="./config/"
OUTPUT_PATH="./output/"
EMG_OUTPUT_PATH="./emgOutput/"
CONFIG_ARRAY=("healthy" "neuro10" "neuro20" "neuro30" "neuro40" "neuro50")

SIM="/Users/kevinsullivan/Desktop/trunk/simtext"
NUM_SIMULATIONS=10

# Run simulation on all datasets
for config in ${CONFIG_ARRAY[@]}; do
    echo $config
    for (( counter=0; counter<$NUM_SIMULATIONS; counter++)); do
        echo "Simulating $counter"
        $SIM -configuration-dir=$CONFIG_PATH$config <<< 'r'
    done
done


# Move emg output to output folder
for config in ${CONFIG_ARRAY[@]}; do
    for (( counter=0; counter<$NUM_SIMULATIONS; counter++)); do
        printf -v FILE_COUNT "%03d" $counter
        EMG=$OUTPUT_PATH$config"/run00$counter/patient/emg/emg1.dat"
        if [ -f $EMG ]; 
        then
        mv $EMG "$EMG_OUTPUT_PATH/sims$NUM_SIMULATIONS/$config/$config-$FILE_COUNT.dat"
        
        else echo $EMG does not exist; fi
    done
done

for config in ${CONFIG_ARRAY[@]}; do
    for (( counter=0; counter<$NUM_SIMULATIONS; counter++)); do
        printf -v FILE_COUNT "%03d" $counter
        EMG=$OUTPUT_PATH$config"/run00$counter/patient/emg/emg1.dat"
        if [ -f $EMG ]; 
        then
        echo $EMG "$EMG_OUTPUT_PATH/sims$NUM_SIMULATIONS/$config/$config-$FILE_COUNT.dat"
        else echo $EMG does not exist; fi
    done
done