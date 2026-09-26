# Mini-transformer (v1.0)

> A lightweight, **Multi-Head**  Transformer decoder with a Feed-Forward Network, normalization layers, and mask's all written in native PyTorch, without the help of any pre-existing transformer libraries or functions E.G (`torch.nn.TransformerDecoder`, `torch.nn.Transformer`, `torch.nn.MultiheadAttention` , `torch.nn.functional.scaled_dot_product_attention`). Is uses CET for training with the help of AdamW for the backpropagation and training.

---

## Overview

**mini-transformer-pytorch** is an  character-level decoder-only language model built from scratch using PyTorch. Version 1.0 Introduces bug fixes, normalization and optimization to make the model run faster, and have high probabillity chances.
`
### Key Technical Specs
* **Tokenizer:** Custom character-level vocabulary (`vocab_size = 41`)
* **Embedding Dimension ($d_{model}$):** $32$
* **Attention Heads ($h$):** $4$ (Head dimension $d_k = 8$)
* **Sequence Length (`block_size`):** $32$ characters
* **Attention Mechanism:** Causal Multi-Head Self-Attention with Scaled Dot-Product & Causal Masking
* **Optimizer:** `AdamW` ($\text{lr} = 1\text{e-}3$)
* **Loss Function:** `CrossEntropyLoss` (now without label smoothing)
* **FFN** `self.ff_layer = nn.Sequential(
        nn.Linear(32,4 * emedding_dimensions),
        nn.GELU(),
        nn.Linear(4 * emedding_dimensions ,emedding_dimensions)
    )`
* **Normilization layers**
    
---


