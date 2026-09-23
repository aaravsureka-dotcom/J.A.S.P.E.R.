# JARVIS-v0.3: Character-Level Transformer from Scratch

> A lightweight, decoder-only causal Transformer language model written entirely in native PyTorch. Built from first principles to study attention dynamics, backpropagation, and sequence modeling without reliance on high-level abstractions.

---

## Overview

**JARVIS-v0.3** is an experimental character-level decoder-only language model. The goal of this release is to validate the core mathematical pipeline of Transformer self-attention: mapping discrete tokens into vector spaces, injecting positional context, calculating causal attention scores, and updating weights via gradient descent.

### Key Technical Specs
* **Tokenizer:** Custom character-level mapping (`vocab_size = 41`)
* **Embedding Dimension ($d_{model}$):** $32$
* **Sequence Length (`block_size`):** $32$ characters
* **Attention Mechanism:** Causal Single-Head Self-Attention with Causal Masking
* **Optimizer:** `AdamW` ($\text{lr} = 1\text{e-}3$)
* **Loss Function:** `CrossEntropyLoss` (with label smoothing)

---

## Architectural Pipeline

```text
Input Sequence (e.g., "abcdefg")
        │
        ▼
   Tokenizer (Token IDs: [0, 1, 2, ...])
        │
        ├──► Token Embedding Matrix  [Batch, Seq_Len, 32]
        └──► Positional Embedding   [Batch, Seq_Len, 32]
        │
        ▼
  Summed Vector Representation (X)
        │
        ├──► Query Projection (Q = X @ W_q)
        ├──► Key Projection   (K = X @ W_k)
        └──► Value Projection (V = X @ W_v)
        │
        ▼
  Causal Self-Attention Score:
  Score = Softmax( (Q @ K^T) + Mask ) @ V
        │
        ▼
  Linear Return Projection  [Batch, Seq_Len, Vocab_Size]
        │
        ▼
  CrossEntropyLoss vs Shifted Target Sequence
