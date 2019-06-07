#! /bin/bash
cd "$(dirname "$0")"

DEVICE_INDEX=0
DEVICE_NAME=titanv
MODE=partitioningNone
BINARY=RestrictedFlowGraph_partitioningNone
ITERATIONS=1000
OUTPUT_PERIOD=0

BIN_PATH=../../bin/linux-x64/Release_Console
ITERATIONS_PATH=iterations/bench/
LOG_PATH="/data/ptheywood/RFG/$MODE/$DEVICE_NAME/log"
XML_REPL_PATTERN=/0.xml

reps=( 1 2 3 )

echo "$MODE"


# find the relevant 0.xml files.
array=()
while IFS=  read -r -d $'\0'; do
    array+=("$REPLY")
done < <(find "$ITERATIONS_PATH" -mindepth 2 -name "0.xml"  -print0)

# make the log dir
mkdir -p "$LOG_PATH"

readarray -t sorted_array < <(for a in "${array[@]}"; do echo "$a"; done | sort -n -t '/' -k3)

# Run each thing.
for init_states_file in "${sorted_array[@]}"
do
    pop=${init_states_file/$ITERATIONS_PATH/}
    pop=${pop/$XML_REPL_PATTERN/}
    # echo $init_states_file
    # echo $pop

    for rep in "${reps[@]}"
    do
        logrep="$LOG_PATH/$rep"
        mkdir -p "$logrep"
        logfile="$logrep/$pop.log"
        echo "Running $pop to $logfile"
        # $BIN_PATH/$BINARY "$init_states_file" "$ITERATIONS" "$DEVICE_INDEX" "$OUTPUT_PERIOD" 2>&1 | tee "$logfile"
        $BIN_PATH/$BINARY "$init_states_file" "$ITERATIONS" "$DEVICE_INDEX" "$OUTPUT_PERIOD" 2>&1 > "$logfile"
    done
done

# Spit out the times taken

grep -rni "total proc" "$LOG_PATH" | sort -t " " -k 4 -n 
