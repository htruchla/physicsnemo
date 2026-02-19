#!/bin/bash
LOCAL_DIR="/media/hana/DATA/XAERONET"
HF_OWNER="neashton"
HF_PREFIX="drivaerml"
BASE_URL="https://huggingface.co/datasets/${HF_OWNER}/${HF_PREFIX}/resolve/main"

mkdir -p "$LOCAL_DIR"

download_run_files() {
    local i=$1
    RUN_DIR="run_$i"
    RUN_LOCAL_DIR="$LOCAL_DIR/$RUN_DIR"
    mkdir -p "$RUN_LOCAL_DIR"

    # Download .stl file only
    if [ ! -f "$RUN_LOCAL_DIR/drivaer_$i.stl" ]; then
        wget -q --show-progress "$BASE_URL/$RUN_DIR/drivaer_$i.stl" -O "$RUN_LOCAL_DIR/drivaer_$i.stl"
    else
        echo "Skipping drivaer_$i.stl (exists)"
    fi

    # Download .vtp file only
    if [ ! -f "$RUN_LOCAL_DIR/boundary_$i.vtp" ]; then
        wget -q --show-progress "$BASE_URL/$RUN_DIR/boundary_$i.vtp" -O "$RUN_LOCAL_DIR/boundary_$i.vtp"
    else
        echo "Skipping boundary_$i.vtp (exists)"
    fi
}

for i in $(seq 1 500); do
    download_run_files "$i" &

    if (( $(jobs -r | wc -l) >= 8 )); then
        wait -n
    fi
done

wait
echo "All downloads complete."