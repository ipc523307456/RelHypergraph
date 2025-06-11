for kappa in 3.0
do
    for trial in 0 1 2 3 4
    do
        python generate_data_large.py --graph "WCGNM_n=200_p=0.2_kappa=${kappa}_${trial}" --num_trials 5
    done
done