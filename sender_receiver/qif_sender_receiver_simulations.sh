#!/bin/bash

# set condition
n=10
batch_size=80
range_end=$((n-1))
conditions=( "hebbian" "antihebbian" )
plasticities=( "oja_rate" "oja_trace" )
Js=( 5 -5 )

# limit amount of threads that each Python process can work with
n_threads=8
export OMP_NUM_THREADS=$n_threads
export OPENBLAS_NUM_THREADS=$n_threads
export MKL_NUM_THREADS=$n_threads
export NUMEXPR_NUM_THREADS=$n_threads
export VECLIB_MAXIMUM_THREADS=$n_threads

# execute python scripts in batches of batch_size
for IDX in $(seq 0 $range_end); do
  for p in "${plasticities[@]}"; do
    for c in "${conditions[@]}"; do
      for J in "${Js[@]}"; do

        # python calls
        (
        echo "Starting job #$((IDX+1)) of ${n} jobs for for p = ${p}, c = ${c} and J = ${J}."
        python /home/richard/PycharmProjects/Heterogeneous_Adaptive_SNNs/sender_receiver/qif_stdp_simulation.py $c $p $J $IDX
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