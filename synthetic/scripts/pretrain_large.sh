export CUDA_VISIBLE_DEVICES=0,1

num_samples=100000
training_arguments="training_arguments_1"

for trial in 0 1 2 3 4
do
    graph="WCGNM_n=20_p=0.2_kappa=3.0_${trial}"
    python pretrain_large.py \
        --graph $graph \
        --training_arguments $training_arguments \
        --num_samples $num_samples \
        --trial 0
done
