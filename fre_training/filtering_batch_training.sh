#!/bin/bash

# set condition
batch_size=60
n=10
range_end=$((n-1))
noises=( 1.0 2.0 4.0 8.0 16.0 32.0 64.0 )
frequencies=( 0.01 0.02 0.04 0.08 0.16 0.32 0.64 1.28 )
models=( "fre" "qif" )

# limit amount of threads that each Python process can work with
n_threads=2
export OMP_NUM_THREADS=$n_threads
export OPENBLAS_NUM_THREADS=$n_threads
export MKL_NUM_THREADS=$n_threads
export NUMEXPR_NUM_THREADS=$n_threads
export VECLIB_MAXIMUM_THREADS=$n_threads

# execute python scripts in batches of batch_size
for IDX in $(seq 0 $range_end); do
  for noise in "${noises[@]}"; do
    for f in "${frequencies[@]}"; do
      for node in "${models[@]}"; do

        # python calls
        (
        echo "Starting jobs for model = ${node}, noise = ${noise}, f = ${f}, and rep = ${IDX}."
        python /home/richard/PycharmProjects/Heterogeneous_Adaptive_SNNs/fre_training/filtering_mf_training.py $node $IDX $noise $f
        sleep 1
        ) &

        # batch control
        if [[ $(jobs -r -p | wc -l) -ge $batch_size ]]; then
              wait -n
        fi

      done
    done
  done
done

wait
echo "All jobs finished."