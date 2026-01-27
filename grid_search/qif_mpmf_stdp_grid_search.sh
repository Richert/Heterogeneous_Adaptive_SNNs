#!/bin/bash

# set condition
n=10
batch_size=40
range_end=$((n-1))
synapses=( "exc" "inh" )
stps=( "sd" "sf" )
group="antihebbian"

# limit amount of threads that each Python process can work with
n_threads=2
export OMP_NUM_THREADS=$n_threads
export OPENBLAS_NUM_THREADS=$n_threads
export MKL_NUM_THREADS=$n_threads
export NUMEXPR_NUM_THREADS=$n_threads
export VECLIB_MAXIMUM_THREADS=$n_threads

# execute python scripts in batches of batch_size
for IDX in $(seq 0 $range_end); do
  for syn in "${synapses[@]}"; do
    for stp in "${stps[@]}"; do

        # python calls
        (
        echo "Starting job #$((IDX+1)) of ${n} jobs for for syn = ${syn} and stp = ${stp}."
        python /home/richard/PycharmProjects/Heterogeneous_Adaptive_SNNs/grid_search/qif_mpmf_stdp_simulation.py $group $stp $syn $c $IDX
        sleep 1
        ) &

        # batch control
        if [[ $(jobs -r -p | wc -l) -ge $batch_size ]]; then
              wait -n
        fi

    done
  done
done

wait
echo "All jobs finished."