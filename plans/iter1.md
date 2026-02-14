pls

1. go through all code in mergekit, and write a thorough call
   stack trace

2. I read from a paper:

To consolidate the capabilities of our domain-expert models, we merge their parameters into a single, unified agent. This approach is supported by prior work [Yadav et al., 2023, Lee et al., 2025], which has shown that merging domain-specific models can yield a single model with superior overall performance. The primary challenge is mitigating parameter interference between the experts. To address this, we employ a three-pronged strategy inspired by recent advances: 1) Normalization: We normalize the magnitude of task vectors (τ i = θ i RL − θ SF T ) to balance contributions from different domains. 2) Dropout: Similar to DARE [Yu et al., 2024a], we apply dropout to prune redundant delta parameters. 3) Erase: Inspired by SCE [Wan et al., 2024], we erase parameter elements with minority-direction updates. This fusion strategy constructs a single model that excels in mathematical reasoning, coding, and agentic capabilities, as demonstrated in Figure 8.

pls maximal re-use mergekit and with minimal changes, implement
this strategy in ./longcat_merge and write an implementation.md
to explain:

1. what's determined in mergekit
2. what's determined in the text above
3. what's inferred by u
