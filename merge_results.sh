#!/usr/bin/env bash

rm -rf merged-results
mkdir merged-results

for dir in out-csvs; do
    for file in $(cd $dir && ls *.csv); do
        noseed=${file%-s*}
        name=${noseed/l0-a0-f0-/}
        cat $dir/$file >> merged-results/${name}.csv
    done


    for file in $(cd $dir && ls *.json); do
        noseed=${file%-s*}
        name=${noseed/l0-a0-f0-/}
        jq '.production' $dir/$file >> merged-results/${name}-production.csv
    done

done
