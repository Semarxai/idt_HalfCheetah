#!/bin/bash
cd "$(dirname "$0")"
for pair in "sac05 ppo11" "sac07 ppo13" "sac09 ppo17" "sac10 ppo27" "sac11 ppo30" "sac13 ppo33"; do
  set -- $pair; [ -f readings/readings_$2.csv.gz ] || python3 post_batch.py $1 $2 >> logs/post.txt 2>&1
done
while pgrep -f "batch.sh 5" > /dev/null; do sleep 20; done
for pair in "sac14 sac18" "ppo08 sac19" "ppo36"; do python3 post_batch.py $pair >> logs/post.txt 2>&1; done
echo "post all done $(date -u)" >> logs/post.txt
