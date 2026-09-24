# Mini-transformer (v0.4)

> A lightweight, **Multi-Head** causal Transformer decoder written from scratch in native PyTorch. Built from first principles to study tensor geometry, multi-head attention dynamics, and sequence modeling without relying on high-level library abstractions.

---

## Overview

**mini-transformer-pytorch** is an experimental character-level decoder-only language model. Version **v0.4** upgrades the self-attention architecture from a single head to **Multi-Head Self-Attention** ($h=4$), allowing the model to jointly process information from multiple representation subspaces simultaneously.

### Key Technical Specs
* **Tokenizer:** Custom character-level vocabulary (`vocab_size = 41`)
* **Embedding Dimension ($d_{model}$):** $32$
* **Attention Heads ($h$):** $4$ (Head dimension $d_k = 8$)
* **Sequence Length (`block_size`):** $32$ characters
* **Attention Mechanism:** Causal Multi-Head Self-Attention with Scaled Dot-Product & Causal Masking
* **Optimizer:** `AdamW` ($\text{lr} = 1\text{e-}3$)
* **Loss Function:** `CrossEntropyLoss` (with label smoothing = 0.1)

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
 Summed Vector Representation (X) [Batch, Seq_Len, 32]
        │
        ├──► Query Projection (Q = X @ W_q)
        ├──► Key Projection   (K = X @ W_k)
        └──► Value Projection (V = X @ W_v)
        │
        ▼
 Multi-Head Reshaping & Transpose:
 [Batch, Seq_Len, 32] ──► [Batch, 4 Heads, Seq_Len, 8 Head_Dim]
        │
        ▼
 Causal Multi-Head Attention Score:
 Score = Softmax( (Q @ K^T) / sqrt(8) + Mask ) @ V
        │
        ▼
 Concatenate Heads & Reshape Back:
 [Batch, 4 Heads, Seq_Len, 8] ──► [Batch, Seq_Len, 32]
        │
        ▼
 Linear Output Projection  [Batch, Seq_Len, Vocab_Size]
```
Training Logs (v0.4 Baseline)
Trained over 15,000 epochs on text corpus using random sequence chunking:

```
LOSS:3.7378 EPOCH:0     | TEXT: rxqqqiqqqsszzzi4szzzszsccssczsqs
LOSS:2.9702 EPOCH:200   | TEXT:    e    e t     t   t           
LOSS:2.8953 EPOCH:1000  | TEXT: tirto  ai  te     antani aan nin
LOSS:2.3357 EPOCH:2000  | TEXT: t n  ntenoinn tn shentnheneteng 
LOSS:2.4831 EPOCH:4000  | TEXT: t ne toesut  re tn ninnin t nano
LOSS:2.4033 EPOCH:6200  | TEXT: a n on tnd tnithatoite t n n  on
LOSS:2.2811 EPOCH:8200  | TEXT: is  resuirys bnirtitdtd intin et
LOSS:2.7162 EPOCH:10000 | TEXT: ng aocae aht   se thetd r aea re
LOSS:2.0139 EPOCH:12800 | TEXT: r tren thur  shaprmtisanddheunht
LOSS:2.0030 EPOCH:14400 | TEXT:  cctend ti cation and testfret a
LOSS:2.3235 EPOCH:14800 | TEXT: uooiis  ng tndni and aerren anoo
```


Project Structure
```

├── corpus.txt          # Training text dataset
├── model.py            # CharacterTransformer PyTorch multi-head architecture
├── train.py            # Dataset sampling & training loop
├── README.md           # Documentation
└── LICENSE             # MIT License
```
