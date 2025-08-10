# Part 5 — Mixture-of-Experts (MoE)

This part focuses purely on the **Mixture-of-Experts feed-forward component**.  
We are **not** implementing self-attention in this module.  
In a full transformer block, the order is generally:

```
[LayerNorm] → [Self-Attention] → [Residual Add]
           → [LayerNorm] → [Feed-Forward Block (Dense or MoE)] → [Residual Add]
```

Our MoE layer is a drop-in replacement for the dense feed-forward block.  
You can imagine it being called **after** the attention output, before the second residual connection.

---

## **5.1 Theory in 60 seconds**
- **Experts**: multiple parallel MLPs; each token activates only a small subset (top-k) → sparse compute.
- **Gate/Router**: scores each token across experts; picks top-k and assigns weights via a softmax.
- **Dispatch/Combine**: send token to chosen experts, run their MLP, combine results using gate weights.
- **Load balancing**: encourage uniform expert usage. A common aux loss (Switch Transformer) is  
  `L_aux = E * Σ ( importance * load )` where:
  - *importance* = avg gate probability per expert  
  - *load* = fraction of tokens routed as primary to that expert

---

## **5.3 Distributed notes (single-GPU friendly)**
- Real MoE implementations distribute experts across GPUs (**expert parallelism**).
- Here we keep everything **on one device** for simplicity. Dispatch is simulated with indexing/masking.
- In production, dispatch/combination typically involves **all-to-all communication** across devices.

---

## **5.4 Hybrid architectures**
- MoE need not replace every FFN — use it in alternating layers or blend outputs:  
  `y = α * Dense(x) + (1 − α) * MoE(x)`

---

**Milestone for this part:** integrate this MoE layer in place of a dense feed-forward in a transformer block and compare efficiency/accuracy trade-offs — even with a toy attention block if you wish.
