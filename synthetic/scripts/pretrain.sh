export CUDA_VISIBLE_DEVICES=0,1

num_samples=100000
training_arguments="training_arguments_2"

for trial in 0
do
    graph="WCGNM_n=100_p=0.2_kappa=3.0_${trial}"
    python pretrain.py \
        --graph $graph \
        --training_arguments $training_arguments \
        --num_samples $num_samples \
        --trial 0
done
