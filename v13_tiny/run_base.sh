#!/bin/zsh
# Base training with resume-on-crash (train.py resumes from runs/base/latest.pt); at most 6 restarts.
cd "$(dirname "$0")"
export TOKENIZERS_PARALLELISM=false
for i in 1 2 3 4 5 6; do
  /opt/anaconda3/bin/python3 train.py --run runs/base --hours 11 && exit 0
  echo "$(date) train.py exited with $?; restart $i in 30 s" >> runs/base/restarts.log
  sleep 30
done
