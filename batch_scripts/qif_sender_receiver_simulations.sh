#!/bin/bash

# set condition
n=10
batch_size=40
range_end=$((n-1))

# limit amount of threads that each Python process can work with
n_threads=8
export OMP_NUM_THREADS=$n_threads
export OPENBLAS_NUM_THREADS=$n_threads
export MKL_NUM_THREADS=$n_threads
export NUMEXPR_NUM_THREADS=$n_threads
export VECLIB_MAXIMUM_THREADS=$n_threads

# execute python scripts in batches of batch_size
for IDX in $(seq 0 $range_end); do

  # python calls
  (
  echo "Starting job #$((IDX+1)) of ${n} jobs."
  python /home/richard/PycharmProjects/Heterogeneous_Adaptive_SNNs/qif_simulations/qif_weight_simulation.py "hebbian" 5 $IDX
  python /home/richard/PycharmProjects/Heterogeneous_Adaptive_SNNs/qif_simulations/qif_weight_simulation.py "antihebbian" 5 $IDX
  python /home/richard/PycharmProjects/Heterogeneous_Adaptive_SNNs/qif_simulations/qif_weight_simulation.py "hebbian" -5 $IDX
  python /home/richard/PycharmProjects/Heterogeneous_Adaptive_SNNs/qif_simulations/qif_weight_simulation.py "antihebbian" -5 $IDX
  sleep 1
  ) &

  # batch control
  if [[ $(jobs -r -p | wc -l) -ge $batch_size ]]; then
        wait -n
  fi

done

wait
echo "All jobs finished."