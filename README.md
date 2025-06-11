# Relational Learning in Pre-Trained Models:  A Theory from Hypergraph Recovery Perspective (ICML 2025)



![Alt text](/Users/yangchen/Desktop/Concept Learning/experiments/rel_hypegraph/framework.png)

## Abstract

Foundation Models (FMs) have demonstrated re- markable insights into the relational dynamics of the world, leading to the crucial question: *how do these models acquire an understanding of world hybrid relations?* Traditional statistical learning, particularly for prediction problems, may over-look the rich and inherently structured informa- tion from the data, especially regarding the rela- tionships between objects. We introduce a mathe- matical model that formalizes relational learning as hypergraph recovery to study pre-training of FMs. In our framework, the world is represented as a hypergraph, with data abstracted as random samples from hyperedges. We theoretically exam- ine the feasibility of a Pre-Trained Model (PTM) to recover this hypergraph and analyze the data efficiency in a minimax near-optimal style. By integrating rich graph theories into the realm of PTMs, our mathematical framework offers pow- erful tools for an in-depth understanding of pre- training from a unique perspective and can be used under various scenarios. As an example, we extend the framework to entity alignment in multimodal learning.



**Paper Link**: https://arxiv.org/abs/2406.11249

**Code**:

- ./synthetic: synthetic experiments.
- ./llm_graph: real-world relation evaluation with LLMs and generate relation graphs.
- ./llm_hypergraph: real-world relation evaluation with LLMs and generate relation hypergraphs