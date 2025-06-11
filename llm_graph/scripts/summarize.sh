d=3
w=2
pt="direct"

for src in "cake" "dog" "fly" "human" "jacket" "orange" "paper" "sea" "table" "zoo"
do
    python summarize.py --source $src --max_depth $d --max_width $w --prompt_type $pt --color "lightblue"
done