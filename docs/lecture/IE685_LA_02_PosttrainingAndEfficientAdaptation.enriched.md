---
document_type: enriched-markdown
source_pdf: IE685_LA_02_PosttrainingAndEfficientAdaptation.pdf
page_count: 88
image_count: 22
extraction_tool: enrich_pdf_local.py
---

# Post-Training and Efficient Adaptation

### IE685 Large Language Models and Agents

---

# Outline

- **Recap: Pre-training Language Models**
  - Scaling up and Emergent Abilities of LLMs
  - Post-Training
    - Instruction Tuning
    - Reinforcement Learning from Human Feedback
  - Efficient Adaptation
    - Adapter-based Tuning
    - LoRa & QLoRa

---

# Recap: Language Models over Time

- Simple n-gram models followed by shallow neural methods and RNNs
- The Transformer architecture started the age of pre-trained language models
  - Large-scale Pre-training followed by task-specific fine-tuning
    - ➔ Transfer Learning

---

# Recap: Pre-training Decoder-only

*Original Sentence:* [Content appears to reference an image/diagram]

---

# Recap: Pre-train/Fine-tune Paradigm of PLMs

- The pre-training stage lets language models learn generic representations and knowledge from **large** corpora, but they are not fine-tuned on any form of user tasks.
- To adapt language models to a specific downstream task, use **comparably small** task-specific datasets for fine-tuning
  - ➔ Transfer knowledge from pre-training, show the model what we want the output to look like and subsequently perform well on **one** task

---

# Scaling up Language Models

- Scaling in three dimensions has been shown to strongly increase task solving capability and generalization:
  - **Model size** in terms of parameters
  - Increasing pre-training **data**
  - Available training **compute**

> Brown, T., Mann, B., Ryder, N., et al. 2020. Language Models are Few-Shot Learners. *Advances in Neural Information Processing Systems*, *33*, pp.1877-1901.

---

# Language Modeling ≠ Solving Tasks

- Language modelling with **next token prediction** does not make the model a competent task solver
- How to adapt to correctly solving tasks?

> Ouyang, L et al., 2022. Training Language Models to follow Instructions with Human Feedback. *Advances in Neural Information Processing Systems*, *35*, pp.27730-27744.

---

# Missing Alignment

- Pre-trained language models are not aligned
- **Objective mismatch**
  - Pre-training is to predict the next word in a sentence
  - Does not involve understanding human intent/values
- **Training data bias**
  - Text from the internet can contain biased, harmful or misleading information
  - LMs don't distinguish between good and bad behavior in training data
- **(Over-)generalization issues**
  - LMs generalization can lead to outputs that are inappropriate in specific contexts
  - Might not align with intended ethics/honesty standard

---

# Emergent Abilities of LLMs

- "Abilities that are not present in small models but arise in large models"

> J. Wei et al., "Emergent Abilities of Large Language Models," CoRR, vol. abs/2206.07682, 2022

- Three typical emergent abilities:
  - **In-context learning**: After providing the LLM with one or several task demonstrations in the prompt, it can generate the expected output (next week)
  - **Instruction following**: Fine-tuning the model with instructions for various tasks at once leads to strong performance on unseen tasks (instruction tuning → our focus today)
  - **Step-by-step reasoning**: LLMs can perform complex tasks by breaking down a problem into smaller steps. The chain-of-thought prompting mechanism is a popular example (next week)

---

# Emergent Abilities of LLMs (Continued)

- Emergent abilities can lead to sudden leaps in performance on various tasks

> J. Wei et al., "Emergent Abilities of Large Language Models," CoRR, vol. abs/2206.07682, 2022

---

# General LLM Training Procedure

### Post-Training

Our Focus in this course (today!)

---

# General LLM Training Procedure

```
Pre-trained Model → Instruction Tuning → RLHF → Aligned Model
"Basic knowledge"   "Follow instructions"   "Human preferences"
```

---

# Overview: The Post-training Stack

- **Instruction-Tuning** (supervised fine-tuning)
  - Curated instruction data (high quality question and answer pairs)
- **RL from Human Feedback**: optimize for human preference signals (style, helpfulness, safety)
  - Preference data: pairs of responses annotated by
    - style/safety/helpfulness
- **RL from Verifiable Rewards**: optimize for math reasoning, code generation, tool use (covered in lecture in two weeks)
  - Math/code/tool use tasks with potentially various solutions but automatically verifiable result
- These stages are complementary and can be mixed!

---

# Instruction Tuning

```
Pre-trained Model → Instruction Tuned Model
"Basic knowledge"   "Follow instructions"
```

---

# Recall: Fine-tuning of PLMs

[Diagram reference: Pre-training and fine-tuning paradigm illustration]

---

# Instruction Tuning

- Leverage emergent ability of the models
- Incorporate instructions into the fine-tuning procedure by prepending a "description" of each task to be carried out
- ➔ Fine-tune pre-trained LMs to map instructions to their corresponding responses.

Example prompt:
> "Please answer the following question..."

---

# Instruction Tuning

- Fine-tune on many tasks at once
- Teaches language model to follow different natural language instructions, so that it can perform well on downstream tasks and even **generalize** to unseen tasks

---

# Instruction Tuning: Adding Diversity

- There is a gap between NLP tasks and user needs…
- More diversity needs to be added to the data...

---

# Adding Diversity via Task Prompts

- Example Task: Summarization
- Create diversity from the **same example** via prompt variations

---

# T0 – An Instruction-tuned LLM

> Sanh, V. et al., Multitask Prompted Training Enables Zero-Shot Task Generalization. In *International Conference on Learning Representations*.

---

# T0 Training Sets

- Collected from multiple public NLP datasets and variety of tasks

> Sanh, V. et al., Multitask Prompted Training Enables Zero-Shot Task Generalization. In *International Conference on Learning Representations*.

---

# Training Mixtures and Unseen Sets

- **Training Mixtures:**
  - Question answering, structure-to-text, summarization
  - Sentiment analysis, topic classification, paraphrase identification
- **Unseen test set:**
  - Sentence completion, BIG-Bench
  - Natural language inference, coreference resolution, word sense disambiguation
- T0 is trained using the T5 transformer (11B model)

> Sanh, V. et al., Multitask Prompted Training Enables Zero-Shot Task Generalization. In *International Conference on Learning Representations*.

---

# Task Adaptation with Prompt Templates

- Instead of directly using input/output pairs, specific instructions are added to explain each task
- The outputs are natural language tokens instead of class labels

> Sanh, V. et al., Multitask Prompted Training Enables Zero-Shot Task Generalization. In *International Conference on Learning Representations*.

---

# Performance on Unseen Tasks

- For T5 and T0, each dot represents one evaluation prompt

> Sanh, V. et al., Multitask Prompted Training Enables Zero-Shot Task Generalization. In *International Conference on Learning Representations*.

---

# Effect of Prompt Variations

- Increasing the number of paraphrasing prompts generally leads to better performance

> Sanh, V. et al., Multitask Prompted Training Enables Zero-Shot Task Generalization. In *International Conference on Learning Representations*.

---

# Effects of More Training Datasets

- Adding more datasets consistently leads to higher median performance

> Sanh, V. et al., Multitask Prompted Training Enables Zero-Shot Task Generalization. In *International Conference on Learning Representations*.

---

# The Effect of Instruction Tuning

[Diagram reference: Comparison showing instruction tuning effectiveness]

---

# Instruction Tuning: Data Collection

### Where to get data for instruction tuning?

- Public NLP datasets (see T0 example above)
- Crowdsourcing data from human annotators
  > Mishra, S. et al., 2022, May. Cross-Task Generalization via Natural Language Crowdsourcing Instructions. In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)* (pp. 3470-3487).
- Generating data with LLMs
  > Zhou, C., et al., 2024. Lima: Less is More for Alignment. *Advances in Neural Information Processing Systems*, *36*.
- Mix of the above
  > Wang, Y., et al., 2023. How Far Can Camels Go? Exploring the State of Instruction Tuning on Open Resources. *Advances in Neural Information Processing Systems*, *36*, pp.74764-74786.

---

# Instruction Tuning: Limitations

- Difficult to collect diverse data
- Resulting models may not be good at open-ended generation tasks
  - Word-by-word learning is incentivized → the resulting LMs generality/creativity is bounded by that of their supervision data
- Resulting models may hallucinate more regularly
  - Labeled data is collected agnostic to the LMs knowledge
  - There may be a mismatch between labelled data and LM knowledge
  - ➔ This may encourage further hallucinations in downstream tasks

---

# Summary: Instruction Tuning

- Instruction tuning enables language models to follow **novel** user instructions that are not seen during fine-tuning
  - ➔ This is what users want!
- Instruction-tuned models perform well on many tasks not just a single one as with task-specific fine-tuning
- **Limitations:**
  - Data collection is expensive, especially for complex tasks (quality and diversity control are necessary)
  - Many tasks do not have a single acceptable output (format) but many can be considered correct
  - Instruction tuning does not directly model **human preferences**

---

# A Glimpse into the Future…

- Instruction tuning today also usually incorporates some form of teaching "**tool use**" to the LLMs
  - *Toolformer* is one of the first papers presenting an instruction tuning dataset specifically for tool use
  - We will learn more about tool using LLMs (and agents) in the lecture in two weeks!

> Schick, T., et al., 2024. Toolformer: Language models can teach themselves to use tools. *Advances in Neural Information Processing Systems*, *36*.

---

# RL from (Human) Feedback

```
Pre-trained Model → RLHF → Aligned Model
"Basic knowledge"   "Human preferences"
```

---

# The Problem of Supervised Fine-tuning

- There is still a misalignment between the ML objective (maximizing the likelihood of a specific piece of human-written text) and what humans actually want (generation of high-quality outputs as determined by humans)
- In this example, the output is not what the user wants and may actually feel hurtful

---

# The Problem of Supervised Fine-tuning

- Language models go through another phase of learning, where they learn how to present information to users and align to human preferences, e.g.:
  - Helpfulness
  - Honesty
  - Harmlessness
- Do you see a problem with these preferences?

---

# Preference Tuning

- We need a way to "train" the model to generate helpful, honest and harmless (HHH) answers
- For this, we need examples
- **Idea:** Collect **preference pairs** and train the model on them
- **Question:** Why preference pairs? Why not fine-tune on HHH question-answer pairs like in instruction tuning?

---

# Preference Tuning

- LLM provides two options for next response
- Human rates better response
- Scoring: Likert Scale – Ranking
- A lot of work to do for humans at scale… Can we (partly) automate this?

---

# RLHF: Intuition

- **Actions:** generating tokens/responses
- **Rewards:** whether humans liked the generation
- **Observations:** conversation with human

---

# RLHF: Intuition

- An agent interacts with an environment by taking actions
- The environment returns a reward for the action and a new state
- Agent uses a policy function to choose an action at a given state
- We need to figure out: (1) reward function and (2) the policy function

---

# Preference Tuning with RLHF

- Imagine a reward function for any output *s* to a prompt
- The reward is higher when humans prefer the output
- Good generation is equivalent to finding reward-maximizing outputs
- What we need to do:
  - Estimate the reward function
  - Find the best generative model that maximizes the expected reward

---

# Step 1: Estimating the Reward

[Diagram reference: Reward model training]

---

# Step 2: Optimizing the Policy Function

- **Policy function:** The model that makes decisions (here: generates responses)
- How do we change our LM parameters to maximize this?

---

# Policy Gradient

- How do we change our LM parameters to maximize this?
- Let's try doing gradient ascent!
- With a bit of math, this can be approximated as Monte Carlo samples from
- This is policy gradient, an approach for estimating and optimizing this objective

> Williams, R.J., 1992. Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine learning*, *8*(3), pp.229-256.

---

# Policy Gradient

- This gives us the following update rule:
  [Mathematical formula reference]

---

# Putting it Together

- First collect a dataset of human preferences
  - Present multiple outputs to human annotators and ask them to rank the output based on preferability

---

# Putting it Together

- Using this data, we can train a reward model
  - The reward model returns a scalar reward which should numerically represent the human preference.

---

# Putting it Together

- We want to learn a policy (a Language Model) that optimizes against the reward model

---

# Putting it Together

- Periodically train the reward model with more samples and human feedback

---

# One Missing Ingredient

- Turns out that this approach does not quite work. (Any guesses why?)

---

# One Missing Ingredient

- Turns out that this approach does not quite work.
  - The policy will learn to "cheat".

---

# One Missing Ingredient

- Will learn to produce an output that would get high reward but is gibberish or irrelevant to prompt.

---

# Regularizing with Pre-trained Model

- Solution: add a penalty term that penalizes too much deviation from the distribution of the pre-trained LM
- This prevents the policy model from diverging too far from pre-trained model
- The presented method is similar to "PPO": Proximal Policy Optimization, one of the main RLHF policies

---

# Comparison with Baselines

- RLHF models are more preferred by human labelers

> Ouyang, L., Wu, J., Jiang, X., et al. 2022. Training Language Models to follow Instructions with Human Feedback. *Advances in Neural Information Processing Systems*, *35*, pp.27730-27744.

---

# Evaluations on Different Aspects

[Diagram reference: Evaluation results from Ouyang et al. 2022]

> Ouyang, L., Wu, J., Jiang, X., et al. 2022. Training Language Models to follow Instructions with Human Feedback. *Advances in Neural Information Processing Systems*, *35*, pp.27730-27744.

---

# The Effect of RLHF

[Diagram reference: RLHF effectiveness illustration]

---

# Limitations of PPO Methods

- Need to train multiple models
- Needs sampling from Language model during fine-tuning
- Complicated reinforcement learning training process
- Is it possible to directly train a language model from human preference annotations?

---

# Direct Preference Optimization

- Removes the iterative reinforcement learning process by directly tuning the model on human preferences
- DPO eliminates the need to:
  - train a reward model
  - sample from the LM during fine-tuning
  - perform large hyperparameter search

---

# DPO versus Baselines

- DPO provides higher expected reward compared to PPO (left)
- Higher win-rate compared to human-written summarizations, evaluated by GPT4 (right)

> Rafailov, R., Sharma, A., Mitchell, E., et al., 2023. Direct Preference Optimization: Your Language Model is Secretly a Reward Model. *Advances in Neural Information Processing Systems*, *36*, pp.53728-53741.

---

# Comparison between PPO and DPO

| Aspect | PPO | DPO |
|--------|-----|-----|
| Process | Complex reinforcement learning, iterative | Simpler fine-tuning by directly fitting reward model |
| Training | More expensive and less stable | Cheaper and more stable |
| Feedback | Can handle more informative human feedback (e.g. numerical ratings) | Can only handle binary signals |

---

# What do people actually use?

[Source reference: Industry survey]

---

# Additional RLHF Methods

- Many more variants exist today...

> Srivastava, S.S. and Aggarwal, V., 2025. A Technical Survey of Reinforcement Learning Techniques for Large Language Models. *arXiv preprint arXiv:2507.04136*.

---

# Open Issues with RLHF

- There remain challenges within each of the three steps:
  - Human feedback
  - Reward model
  - Policy

> Casper, S., et al., 2023. Open Problems and Fundamental Limitations of Reinforcement Learning from Human Feedback. *Transactions on Machine Learning Research*.

---

# Challenges: Human Feedback

- **Biases of human evaluators**
  - Studies found that ChatGPT became politically biased after RLHF
- **Good oversight is difficult**
  - Evaluators are paid per example and may make mistakes given time constraints
  - Poor feedback when evaluating difficult tasks
- **Data Quality**
  - Cost/Quality tradeoff
- **Tradeoff between richness and efficiency of feedback types**
  - Comparison-based feedback, scalar feedback, correction feedback, language feedback, …

---

# Challenges: Reward Model

- A single reward model cannot represent a diverse society of humans
- Reward misgeneralization: reward model may fit with human preference data due to unexpected features
- Evaluation of reward model is difficult and expensive

---

# Challenges: Policy

- Robust reinforcement learning is difficult
  - Balance between exploring new actions and exploiting known rewards
  - Challenge increases in high-dimensional or sparse reward settings
- Policy misgeneralization: training and deployment environments are different

---

# Summary: RLHF

- Reinforcement Learning from Human Feedback allows to directly model human preferences and generalize beyond the labelled data
- Reinforcement Learning from Human Feedback can improve on doing only instruction-tuning
- Tricky to get right
- "Alignment Tax": performance on tasks may suffer in favour of modelling outputs to human preference

---

# Summary: RLHF (Continued)

- Human preferences are unreliable!
  - "Reward hacking" is common problem in RL
  - Chatbots are rewarded to produce responses that are authoritative and helpful, **regardless of truth**, which can result in **hallucinations**
- Models of human preferences are even more unreliable!
- Still very data expensive
- Very underexplored and fast-moving research area

---

# Outlook: Reasoning LLMs

- Focus since 2025 is on Reasoning LLMs (OpenAI GPT 5.2, Claude Opus 4.6, Kimi K2.5, etc.)
  - Incorporation of chain-of-thought prompting (next week) into training procedure
  - Introduction of additional thinking tokens during inference
  - Reinforcement learning is used to automatically generate reasoning examples for training (and to learn correct tool usage)
    - How to verify the final output is correct if we do not have labels?
    - → Use domains where correct answer can be programmatically derived (math, coding, ...)
- We will talk about tool-using Reasoning LLMs, their training, inference, and LLM agents in the lecture in two weeks!

**References:**
- OpenAI Blog: https://openai.com/index/learning-to-reason-with-llms/
- OpenAI o1 system card: https://cdn.openai.com/o1-system-card-20241205.pdf
- Deepseek R1 paper: https://arxiv.org/abs/2501.12948

---

# Datasets for Post-Training

> Tie, G., Zhao, Z., Song, D., et al. 2025. A Survey on Post-Training of Large Language Models. *arXiv preprint arXiv:2503.06072*.

---

# Parameter-efficient Fine-tuning (PEFT)

- Fine-tuning all parameters is **impractical**, especially with LLMs
- **Solution:** Tune only parts of the parameters
- State-of-the-art models are massively overparameterized anyway
- ➔ Parameter-efficient fine-tuning can match performance of full fine-tuning

---

# Adapter-based Fine-tuning

- When fine-tuning PLMs, we usually add an additional layer to the top of the Transformer
- Adapter modules follow the same principle but add additional **smaller** layers to the original network
- During fine-tuning, the original parameters are **frozen** and only the adapter weights are updated

---

# Comparison to Standard Fine-Tuning

- Adapter-based fine-tuning achieves similar performance to full fine-tuning with orders of magnitude fewer trained parameters

> Houlsby, N., et al., 2019. Parameter-efficient Transfer Learning for NLP. In *International Conference on Machine Learning* (pp. 2790-2799).

---

# Adapter-based Fine-tuning

**Pros:**
- Empirically effective in multi-task settings
- Computationally efficient compared to full fine-tuning

**Cons:**
- Adding new layers makes the model slower during inference time
- Makes the model size larger overall

---

# LoRA: Low-Rank Adaptation

- For each downstream task, we learn a different set of parameters Δφ
  - GPT-3 has |φ| of 175 billion
  - |Δφ| = |φ| for full fine-tuning
  - Expensive and memory inefficient!
- **LoRA key idea:** encode the task specific parameter increment Δφ = Δφ(Θ) by a smaller-sized set of parameters Θ, |Θ| ≪ |φ|

> Hu, E.J., et al., LoRA: Low-Rank Adaptation of Large Language Models. In *International Conference on Learning Representations*.

---

# LoRA: Low-Rank Adaptation

- It was shown that updates to weights have low intrinsic rank during adaptation
- For one weight matrix: constrain the update with a low-rank decomposition
- Only A and B contain **trainable** parameters

---

# LoRA: Low-Rank Adaptation

- With increasing number of trainable parameters, the LoRA training converges to fully training the original model
- **No additional inference latency** when switching to a different task—recover W by subtracting BA and adding different B'A'

---

# LoRA: swap matrices = swap tasks

[Diagram reference: LoRA matrix swapping illustration]

---

# Applying LoRA to Transformers

- In general, applicable to **any** deep learning weight matrix
- For GPT3-175B with r ranging from 2-64:
  - VRAM: 1.2TB → 350GB
  - Checkpoint storage: 350GB → 35MB (**10,000 times smaller**)
- LoRA can outperform several baselines with comparable or fewer trainable parameters

---

# Where to apply LoRA?

- Original paper suggested applying only in multi-head attention layers

> Hu, E.J., et al., LoRA: Low-Rank Adaptation of Large Language Models. In *International Conference on Learning Representations*.

- Today it is recommended to apply LoRA to the Feed Forward layers only, for good results with high efficiency

> Schulman, John and Thinking Machines Lab, "LoRA Without Regret", Thinking Machines Lab: Connectionism, Sep 2025.

- For maximum performance, applying LoRA to all layers is the safest bet, but this costs efficiency.

---

# Comparison of Fine-tuning Methods

- Given enough data and computing resources:
  - Overall performance on T5-base: Full fine-tuning > LoRA > Adapters

> Ding, N., Qin, Y., Yang, G., Wei, F., Yang, Z., Su, Y., Hu, S., Chen, Y., Chan, C.M., Chen, W. and Yi, J., 2022. Delta tuning: A comprehensive study of parameter efficient methods for pre-trained language models. *arXiv preprint arXiv:2203.06904*.

---

# QLoRA

- One of the main bottlenecks in fine-tuning even with LoRA is the GPU memory required to store all model weights
- **QLoRA Idea:** Store frozen weights in quantized form (4-bit) to relieve the bottleneck (but dequantize when needed)

> Dettmers, T., Pagnoni, A., Holtzman, A. and Zettlemoyer, L., 2023. QLoRA: Efficient Finetuning of Quantized LLMs. *Advances in Neural Information Processing Systems*, *36*, pp.10088-10115.

---

# Summary: Efficient Adaptation

| Method | Advantages | Disadvantages |
|--------|------------|---------------|
| **Adapters** | Faster than full model fine-tuning while still achieving similar performance; Can swap out adapters depending on the task | Introduces additional parameters making inference slower |
| **LoRA/QLoRA** | Faster than full model fine-tuning while achieving (close to) the same performance; Straightforward to swap between tasks; No additional parameters, same inference latency | - |

---

# See you next week!

- Next time: Prompt engineering and evaluation of LLMs!
  - Zero-shot, in-context learning, chain-of-thought, …
  - Types of LLM evaluation

---

# Credits

This slide set is based on slides from:
- Jiaxin Huang
- Mrinmaya Sachan
- Tatsunori Hashimoto
- Afshine & Shervine Amidi
- Daniel Khashabi

Many thanks to all of you!