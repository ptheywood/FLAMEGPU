#! /bin/bash
cd "$(dirname "$0")"

INIT_POPS=(
1024
2048
4096
8192
16384
32768
65536
131072
262144
524288
)

INIT_STATES="0.xml"
NETWORK_JSON="network.json"

SED_PATTERN="s/<INIT_POPULATION>512<\/INIT_POPULATION>/<INIT_POPULATION>VALUE<\/INIT_POPULATION>/g"

# For each specified pop size, copy the files and modify 0.xml via sed. 
for pop in "${INIT_POPS[@]}"
do
    local_xml="$pop/$INIT_STATES"
    local_json="$pop/$NETWORK_JSON"
    local_pattern=${SED_PATTERN/VALUE/$pop}
    # Make the dir
    mkdir -p "$pop"
    # Copy xml
    cp $INIT_STATES $local_xml
    # Copy json
    cp $NETWORK_JSON $local_json
    # Modify xml
    sed -i $local_pattern $local_xml
done
