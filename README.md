# mini-transformer-pytorch (v0.3)

> A lightweight, single-head causal Transformer decoder written from scratch in native PyTorch. Built from first principles to study tensor geometry, attention dynamics, and sequence modeling without reliance on high-level library abstractions.

---

## Overview

**mini-transformer-pytorch** is an experimental character-level decoder-only language model. The goal of this release is to validate the core mathematical pipeline of Transformer self-attention: mapping discrete tokens into vector spaces, injecting positional context, calculating causal attention scores, and updating weights via gradient descent.

### Key Technical Specs
* **Tokenizer:** Custom character-level mapping (`vocab_size = 41`)
* **Embedding Dimension ($d_{model}$):** $32$
* **Sequence Length (`block_size`):** $32$ characters
* **Attention Mechanism:** Causal Single-Head Self-Attention with Causal Masking
* **Optimizer:** `AdamW` ($\text{lr} = 1\text{e-}3$)
* **Loss Function:** `CrossEntropyLoss` (with label smoothing)

---

## Architectural Pipeline


## Training Logs (v0.3 Baseline)

Trained over 15,000 epochs on `corpus.txt` using a batch size of 1 with dynamic random chunk sampling:

```text
Epoch 0     | Loss: 3.82 | Text: ,,,,s!f7644g4gg22nn22xsax4agx!cx
Epoch 2000  | Loss: 2.98 | Text: nn    n  a tetaet    o  t tn  n 
Epoch 5000  | Loss: 2.68 | Text:     t  n   et      r     a nat t
Epoch 13800 | Loss: 2.32 | Text: tene te  de  ,nd , , and an, ir 
Epoch 14800 | Loss: 2.95 | Text: tnnhtinii inteten  tepea  ae  a

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
