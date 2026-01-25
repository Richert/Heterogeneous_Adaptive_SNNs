#!/bin/bash

# set condition
n=10
batch_size=60
range_end=$((n-1))
synapses=( "exc" "inh" )
stps=( "sd" "sf" )
tau_ps=( 20.0 40.0 60.0 80.0 100.0 )
tau_ds=( 20.0 40.0 60.0 80.0 100.0 )
a_ps = ( 0.001 0.003 0.005 0.007 0.009 )
a_ds = ( 0.001 0.003 0.005 0.007 0.009 )

# limit amount of threads that each Python process can work with
n_threads=2
export OMP_NUM_THREADS=$n_threads
export OPENBLAS_NUM_THREADS=$n_threads
export MKL_NUM_THREADS=$n_threads
export NUMEXPR_NUM_THREADS=$n_threads
export VECLIB_MAXIMUM_THREADS=$n_threads

# execute python scripts in batches of batch_size
for IDX in $(seq 0 $range_end); do
  for tau_p in "${tau_ps[@]}"; do
    for tau_d in "${tau_ds[@]}"; do
      for a_p in "${a_ps[@]}"; do
        for a_d in "${a_ds[@]}"; do

          # python calls
          (
          echo "Starting job #$((IDX+1)) of ${n} jobs for for p = ${p} and J = ${c}."
          python /home/richard/PycharmProjects/Heterogeneous_Adaptive_SNNs/grid_search/qif_mpmf_stdp_simulation.py $p $tau $c $IDX
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