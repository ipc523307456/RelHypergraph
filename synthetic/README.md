# Synthetic Experiments

Synthetic experiments with BERT.

## Usage

1. Create a conda environment and install required packages.

   ```
   conda create --name [ENV] --file requirements.txt
   conda activate [ENV]
   ```

2. Download BERT and put it in ./hf_models.

   ```
   ./hf_models/bert-base-cased
   ```

​	One can put the BERT model in a different directory and modify the path in the code accordingly.

3. Generate graphs.

   ```
   python generate_STAR_graph.py [ARGS]
   python generate_WCGNM_graph.py [ARGS]
   ```

4. Generate training data.

   ```
   python generate_data.py --graph [PATH_TO_GRAPH] --num_trials [NUM_TRIALS]
   ```

5. Pre-train the model.

   ```
   python pretrain.py \
       --graph [GRAPH] \
       --training_arguments [TRAINING_ARGUMENTS] \
       --num_samples [NUM_SAMPLES] \
       --trial [TRIAL]
   ```

6. Evaluate the model.

   ```
   python evaluate.py \
       --graph [GRAPH] \
       --training_arguments [TRAINING_ARGUMENTS] \
       --trial [TRIAL]
   ```

See ./scripts for examples.