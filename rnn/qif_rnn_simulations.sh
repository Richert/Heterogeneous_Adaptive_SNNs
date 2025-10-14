#!/bin/bash

# set condition
batch_size=10
n=10
range_end=$((n-1))
bs=( 0.1 )
noises=( 0.0 1.0 10.0 )
deltas=( 0.0 0.4 0.8 1.2 1.6 2.0 )

# limit amount of threads that each Python process can work with
n_threads=8
export OMP_NUM_THREADS=$n_threads
export OPENBLAS_NUM_THREADS=$n_threads
export MKL_NUM_THREADS=$n_threads
export NUMEXPR_NUM_THREADS=$n_threads
export VECLIB_MAXIMUM_THREADS=$n_threads

# execute python scripts in batches of batch_size
for IDX in $(seq 0 $range_end); do
  for b in "${bs[@]}"; do
    for noise in "${noises[@]}"; do
      for delta in "${deltas[@]}"; do

          # python calls
          (
          echo "Starting job for noise = ${noise}, delta = ${delta}, b = ${b}, and rep = ${IDX}."
          python /home/richard/PycharmProjects/Heterogeneous_Adaptive_SNNs/rnn/fre_mp_simulation.py $noise $delta $b $IDX
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