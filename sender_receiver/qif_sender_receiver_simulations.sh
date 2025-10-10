#!/bin/bash

# set condition
batch_size=80
range_end=$((n-1))
conditions=( "hebbian" "antihebbian" )
plasticities=( "oja_rate" "oja_trace" )
bs=( 0.0 0.01 0.1 1.0 )
noises=( 0.0 0.5 1.0 2.0 )

# limit amount of threads that each Python process can work with
n_threads=2
export OMP_NUM_THREADS=$n_threads
export OPENBLAS_NUM_THREADS=$n_threads
export MKL_NUM_THREADS=$n_threads
export NUMEXPR_NUM_THREADS=$n_threads
export VECLIB_MAXIMUM_THREADS=$n_threads

# execute python scripts in batches of batch_size
for p in "${plasticities[@]}"; do
  for c in "${conditions[@]}"; do
      for n in "${noises[@]}"; do
        for b in "${bs[@]}"; do

          # python calls
          (
          echo "Starting job for p = ${p}, c = ${c}, n = ${n}, and b = ${b}."
          python /home/richard/PycharmProjects/Heterogeneous_Adaptive_SNNs/sender_receiver/qif_simulation.py $p $c $n $b
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