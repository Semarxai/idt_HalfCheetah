#!/bin/bash
# usage: batch.sh <batch_no> agentA1 agentA2 -- agentB1 agentB2 ...   (two parallel workers)
b=$1; shift; A=(); Bq=(); side=A
for x in "$@"; do if [ "$x" == "--" ]; then side=B; elif [ $side == A ]; then A+=($x); else Bq+=($x); fi; done
cd "$(dirname "$0")"
python3 runner.py long 10 100000 ${A[@]} > logs/batch${b}_w1.txt 2>&1 &
python3 runner.py long 10 100000 ${Bq[@]} > logs/batch${b}_w2.txt 2>&1 &
wait; echo "batch $b finished $(date -u)" >> logs/batches.txt
