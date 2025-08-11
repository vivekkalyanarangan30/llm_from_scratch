# LLM from Scratch — Hands-On Curriculum (PyTorch)

## Part 0 — Foundations & Mindset
- **0.1** Understanding the high-level LLM training pipeline (pretraining → finetuning → alignment)
- **0.2** Hardware & software environment setup (PyTorch, CUDA, mixed precision, profiling tools)

## Part 1 — Core Transformer Architecture
- **1.1** Positional embeddings (absolute learned vs. sinusoidal)
- **1.2** Self-attention from first principles (manual computation with a tiny example)
- **1.3** Building a *single attention head* in PyTorch
- **1.4** Multi-head attention (splitting, concatenation, projections)
- **1.5** Feed-forward networks (MLP layers) — GELU, dimensionality expansion
- **1.5** Residual connections & **LayerNorm**
- **1.6** Stacking into a full Transformer block
- **Milestone:** Minimal GPT-like model that overfits on a toy dataset.

## Part 2 — Training a Tiny LLM
- **2.1** Byte-level tokenization
- **2.2** Dataset batching & shifting for next-token prediction
- **2.3** Cross-entropy loss & label shifting
- **2.4** Training loop from scratch (no Trainer API)
- **2.5** Sampling: temperature, top-k, top-p
- **2.6** Evaluating loss on val set
- **Milestone:** Train on a 100KB text file and generate coherent snippets.

## Part 3 — Modernizing the Architecture
- **3.1** **RMSNorm** (replace LayerNorm, compare gradients & convergence)
- **3.2** **RoPE** (Rotary Positional Embeddings) — theory & code
- **3.3** SwiGLU activations in MLP
- **3.4** KV cache for faster inference
- **3.5** Sliding-window attention & **attention sink**
- **3.6** Rolling buffer KV cache for streaming
- **Milestone:** Benchmarks before vs. after each upgrade.

## Part 4 — Scaling Up
- **4.1** Switching from byte-level to BPE tokenization
- **4.2** Gradient accumulation & mixed precision
- **4.3** Learning rate schedules & warmup
- **4.4** Checkpointing & resuming
- **4.5** Logging & visualization (TensorBoard / wandb)
- **Milestone:** Train a 10–50M param model on a medium dataset.

## Part 5 — Mixture-of-Experts (MoE)
- **5.1** MoE theory: expert routing, gating networks, and load balancing
- **5.2** Implementing MoE layers in PyTorch
- **5.3** Training stability and communication overhead in distributed setups
- **5.4** Combining MoE with dense layers for hybrid architectures
- **Milestone:** MoE-augmented model with measurable efficiency or accuracy gains.

## Part 6 — Supervised Fine-Tuning (SFT)
- **6.1** Instruction dataset formatting (prompt + response)
- **6.2** Causal LM loss with masked labels
- **6.3** Curriculum learning for instruction data
- **6.4** Evaluating outputs against gold responses
- **Milestone:** Your model can follow basic instructions.


## TODOs:
## Part 7 — Reward Modeling
- **7.1** Preference datasets (pairwise rankings)
- **7.2** Reward model architecture (shared transformer encoder)
- **7.3** Loss functions: Bradley–Terry, margin ranking loss
- **7.4** Sanity checks for reward shaping
- **Milestone:** Reward model that can score outputs for helpfulness.

## Part 8 — RLHF, DPO & GRPO
- **8.1** PPO: policy, value head, KL penalty
- **8.2** GRPO: group relative preference optimization
- **8.3** DPO: direct preference optimization
- **8.4** Integrating reward model & generator
- **8.5** Stabilizing training with reward normalization
- **Milestone:** Aligned chatbot with reasoning-friendly outputs.

## Part 9 — Reasoning & Advanced Tricks
- **9.1** Chain-of-thought prompting & self-consistency
- **9.2** Multi-headed latent attention (DeepSeek-style)
- **9.3** Tool use & API calling (synthetic datasets)
- **9.4** Evaluating reasoning compliance
