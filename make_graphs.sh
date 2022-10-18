mkdir out-csvs

for f in model1 model1small model2 model3b model4do10 model4do4; do
    for prior in "--priortype 0" "--priortype 1" "--priortype 2" "--priortype 3" "--priortype 4" "--priortype 0 --perfect"; do
        for threshold in 75 90 100; do
            for seed in $(seq 100); do
            for fast in ""; do
            echo "python3 paper-outputs.py $prior --model $f --threshold $threshold --seed $seed $fast --outdir out-csvs"
done
done
done
done
done
