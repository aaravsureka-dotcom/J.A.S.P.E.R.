# Mini-transformer (v1.0)

> A lightweight, **Multi-Head**  Transformer decoder with a Feed-Forward Network, normalization layers, and mask's all written in native PyTorch, without the help of any pre-existing transformer libraries or functions E.G (`torch.nn.TransformerDecoder`, `torch.nn.Transformer`, `torch.nn.MultiheadAttention` , `torch.nn.functional.scaled_dot_product_attention`). Is uses CET for training with the help of AdamW for the backpropagation and training.

---

## Overview

**Nano-GPT** is an  character-level decoder-only language model built from scratch using PyTorch. Version 1.0 Introduces bug fixes, normalization and optimization to make the model run faster, and have high probabillity chances.
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

**What actually happens**
* Text given from user `Hello world`
* Text converted into numerical representations using the tokenizer ` [7, 4, 11, 11, 14, 36, 22, 14, 17, 11, 3]`
* Numerical representation's given in the model's forward function `model.forward(tokenizer_text)`
* Positional value's added to each 32 numbers of the tokens:
* 1st Normalization layer added `clean_x = self.layer_norm1(x)`
* clean_x goes through the Query Layer, Key layer and value layer
* result splits into 4 heads and goes through them
* each head get's attached back
* scores are calculated using `score = (qlayer @ klayer.transpose(-2, -1)) / math.sqrt(head_size)`
* Casual Mask is applied to make sure the model does not "Cheat" `causal_mask = torch.triu(torch.ones(t, t, dtype=torch.bool, device=inx.device), diagonal=1)`
* Goes through a softmax
* Matrix Multiplied by the values layer
* Goes through another liner layer
* 2nd Normilization layer added ` clean_x = self.layer_norm2(x)`
* Goes through FFN `out_put_final = self.ff_layer(clean_x)`
* Final adding up of X ` x = x + out_put_final`
* X goes through the return layer `self.return_layer(x)`
  


