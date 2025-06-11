# Real-World Relation Hypergraph Evaluation with LLMs

Extract entities and relations from ConceptNet. Generate relation hyergraphs via LLM responses.

## Usage

1. Extract entities and relations from ConceptNet: specify a source entity [SRC] and extract a subgraph by choosing the [MAX_WIDTH] most closely related entities at each depth until reaching depth [MAX_DEPTH].

   ```
   python generate_data.py --source [SRC] --max_depth [MAX_DEPTH] --max_width [MAX_WIDTH]
   python graph.py --source [SRC] --max_depth [MAX_DEPTH] --max_width [MAX_WIDTH]
   ```

2. Generate prompts.

   ```
   python generate_prompt.py --source [SRC] --max_depth [MAX_DEPTH] --max_width [MAX_DEPTH] --prompt_type [PROMPT_TYPE]
   ```

3. Query a LLM and analyze its responses to generate hypergraphs.

   ```
   python evaluate.py --source [SRC] --max_depth [MAX_DEPTH] --max_width [MAX_WIDTH] --prompt_type [PROMPT_TYPE] --model [MODEL]
   python analyze.py --source [SRC] --max_depth [MAX_DEPTH] --max_width [MAX_WIDTH] --prompt_type [PROMPT_TYPE] --model [MODEL]
   python visualize.py --source [SRC] --max_depth [MAX_DEPTH] --max_width [MAX_WIDTH] --prompt_type [PROMPT_TYPE] --model [MODEL]
   ```

See scripts/pipeline.sh for examples.