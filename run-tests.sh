#!/bin/bash

# run the tests in background and in parallel (make sure to have 4 cores!)

OPPONENTS=("type_selector" "pruned_bfs" "minimax" "tuned_tree")

if [ ! -d "outputs" ]; then
    mkdir outputs
fi

for opponent in "${OPPONENTS[@]}"
do
    echo "Running tests against $opponent"
    python test-executor.py $opponent > outputs/$opponent.txt &
done