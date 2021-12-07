#!/bin/bash

# Change DATA_TYPE based on running training or testing
NUM_SIMULATIONS=$1
DATA_TYPE=$2

CONFIG_PATH="./config"
OUTPUT_PATH="./output/sims$NUM_SIMULATIONS"
EMG_OUTPUT_PATH="./emgOutput"
CONFIG_ARRAY=("healthy" "neuro10" "neuro20" "neuro30" "neuro40" "neuro50")

SIM="/Users/kevinsullivan/Desktop/trunk/simtext"

# Run simulation on all datasets
for config in ${CONFIG_ARRAY[@]}; do
    echo $config
    for (( counter=0; counter<$NUM_SIMULATIONS; counter++)); do
        echo "Simulating $counter"
        ($SIM -configuration-dir=$CONFIG_PATH/$config <<< 'r') > /dev/null 2>&1
    done
done

# Move emg output to output folder
mkdir -p $OUTPUT_PATH
mkdir -p $EMG_OUTPUT_PATH/sims$NUM_SIMULATIONS
mkdir -p $EMG_OUTPUT_PATH/sims$NUM_SIMULATIONS/$DATA_TYPE
for config in ${CONFIG_ARRAY[@]}; do
    for (( counter=0; counter<$NUM_SIMULATIONS; counter++)); do
        printf -v FILE_COUNT "%03d" $counter
        EMG=$CONFIG_PATH/$config"/run00$counter/patient/emg/emg1.dat"
        if [ -f $EMG ]; 
        then
        mv $EMG "$EMG_OUTPUT_PATH/sims$NUM_SIMULATIONS/$DATA_TYPE/$config-$FILE_COUNT.dat"
        
        else echo $EMG does not exist; fi
    done
done

# Delete contents of output folder and move log folders

for config in ${CONFIG_ARRAY[@]}; do
    echo Removing simulations for $config
    rm -rf $CONFIG_PATH/$config/run*
    rm -rf $CONFIG_PATH/$config/simulator.log
done


if [ ! -d $EMG_OUTPUT_PATH/sims$NUM_SIMULATIONS ]; then echo "no"; fi