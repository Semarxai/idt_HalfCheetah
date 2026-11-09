#!/bin/bash
cd "$(dirname "$0")"
while pgrep -f "batch.sh 2" > /dev/null; do sleep 20; done
./batch.sh 3 sac09 ppo17 -- sac10 ppo27
./batch.sh 4 sac11 ppo30 -- sac13 ppo33
./batch.sh 5 sac14 sac18 ppo08 -- sac19 ppo36
