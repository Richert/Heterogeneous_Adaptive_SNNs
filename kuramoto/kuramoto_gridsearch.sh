#!/bin/bash

# set condition
batch_size=5
deltas=( 0.01 0.03 0.1 0.3 1.0 3.0 )
ds=( 10 50 100 )
mus=( 0.001 0.003 0.01 0.03 0.1 0.3 1.0 )
distribution=( "lorentzian" "uniform" )

# limit amount of threads that each Python process can work with
n_threads=15
export OMP_NUM_THREADS=$n_threads
export OPENBLAS_NUM_THREADS=$n_threads
export MKL_NUM_THREADS=$n_threads
export NUMEXPR_NUM_THREADS=$n_threads
export VECLIB_MAXIMUM_THREADS=$n_threads

# execute python scripts in batches of batch_size
for dist in "${distribution[@]}"; do
  for d in "${ds[@]}"; do
    for delta in "${deltas[@]}"; do
      for mu in "${mus[@]}"; do

        # python calls
        (
        echo "Starting job #$((IDX+1)) of ${n} jobs for distribution = ${dist}, d = ${d}, Delta = ${delta}, and mu = ${mu}."
        python /home/richard/PycharmProjects/Heterogeneous_Adaptive_SNNs/kuramoto/kuramoto_gridsearch.py --d $d --Delta0 $delta --mu $mu --dist $dist
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