d=3
w=2
pt="direct"

for src in "cake" "dog" "fly" "human" "jacket" "orange" "paper" "sea" "table" "zoo"
do 
    echo $src
    python generate_data.py --source $src --max_depth $d --max_width $w
    python graph.py --source $src --max_depth $d --max_width $w
    python generate_prompt.py --source $src --max_depth $d --max_width $w --prompt_type $pt

    for model in "gpt-3.5-turbo-0613" "gpt-4-0613" 
    do
        echo $model
        python evaluate.py --source $src --max_depth $d --max_width $w --prompt_type $pt --model $model
        python analyze.py --source $src --max_depth $d --max_width $w --prompt_type $pt --model $model
        python visualize.py --source $src --max_depth $d --max_width $w --prompt_type $pt --model $model
    done
done