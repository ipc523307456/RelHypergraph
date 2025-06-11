export CUDA_VISIBLE_DEVICES=0
export QT_QPA_PLATFORM=offscreen

training_arguments="training_arguments_1"

trial=0
for kappa in 3.0
do
    for graph_id in 0 1 2 3 4
    do
        graph="WCGNM_n=100_p=0.2_kappa=${kappa}_${graph_id}"
        python evaluate.py \
            --graph $graph \
            --training_arguments $training_arguments \
            --trial $trial
    done
done