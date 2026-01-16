# Conditional Trigger Model Conditioning Mechanisms

This document overviews the implementation of class-conditional trigger generation in the `bdc_*` family of models. These models are designed to generate specific backdoor triggers based on a target class label, enabling multi-target or "all-to-all" attacks.

## Core Concept
All conditional models follow a similar pattern:
1.  **Label Embedding**: Map the target class integer ($c$) to a dense vector representation ($\phi(c)$).
2.  **Projection/Broadcast**: Project or reshape the embedding to match the dimensions of the intermediate feature maps of the generator network.
3.  **Additive Conditioning**: Add the conditioning vector to the feature map at a strategic point in the network.

---

## 1. CNN-based Generator (`bdc_cnn.py`)

The `bdc_CNN` model introduces conditioning into a standard 1D CNN architecture.

### Conditioning Mechanism
*   **Embedding**: `nn.Embedding(num_class, 64)`
*   **Projection**: A linear layer projects the 64-dim embedding to match the hidden dimension ($256 \times D$).
    *   `self.label_proj = nn.Linear(64, 256 * D)`
*   **Injection Point**: The conditioning is added between the two Fully Connected (FC) layers, after the convolutional feature extraction.
*   **Process**:
    1.  Embed label: $e = \text{Embedding}(y)$
    2.  Project: $h_{cond} = \text{Linear}(e)$
    3.  Extract features: $h_{conv} = \text{ReLU}(\text{FC}_1(\text{ReLU}(\text{Conv2}(\dots))))$
    4.  **Inject**: $h_{combined} = h_{conv} + h_{cond}$ (Broadcasting across time dimension)
    5.  Output: $\text{Trigger} = \text{Tanh}(\text{FC}_2(h_{combined}))$

```python
# Code Snippet from bdc_cnn.py
label_emb = self.label_embedding(bd_labels)
label_cond = self.label_proj(label_emb).unsqueeze(1) # [B, 1, 256*D]
...
x = self.fc1(x) # [B, L, 256*D]
x = x + label_cond # Additive conditioning
x = self.fc2(x)
```

---

## 2. PatchTST-based Generator (`bdc_patchTST.py`)

The `bdc_PatchTST` model adapts the PatchTST architecture for conditional generation.

### Conditioning Mechanism
*   **Embedding**: `nn.Embedding(num_class, d_model_bd)`
*   **Injection Point**: The conditioning is added directly to the **patch embeddings** before they enter the Transformer Encoder. This serves as a "class token" bias for every patch.
*   **Process**:
    1.  Embed label: $e = \text{Embedding}(y)$
    2.  Patch Input: $x_{enc} \rightarrow \text{PatchEmbedding}(x_{enc})$
    3.  **Inject**: The label embedding is broadcasted across all variables ($N$) and added to the patch embeddings.
        *   $Z_{patches} = \text{PatchEmbed}(x) + \phi(c)$
    4.  Encode: $Z_{out} = \text{TransformerEncoder}(Z_{patches})$
    5.  Decode: Flatten head projects back to time series dimension.

```python
# Code Snippet from bdc_patchTST.py
label_emb = self.label_embedding(bd_labels) # [B, d_model]
...
enc_out, n_vars = self.patch_embedding(x_enc)
# Reshape and add
label_cond = label_emb.unsqueeze(1).repeat(1, n_vars, 1).reshape(B * n_vars, 1, d_model)
enc_out = enc_out + label_cond
```

---

## 3. TimesNet-based Generator (`bdc_TimesNet.py`)

The `bdc_TimesNet` model conditions the generation process within the specialized `TimesBlock` layers.

### Conditioning Mechanism
*   **Embedding**: `nn.Embedding(num_class, d_model_bd)`
*   **Injection Point**: The embedding is passed as an auxiliary input to **every** `TimesBlock` in the stack.
*   **Process**:
    1.  Embed label: $c_{emb} = \text{Embedding}(y)$
    2.  Expand: $c_{cond} = c_{emb}.\text{unsqueeze}(1)$
    3.  Iterative Block Processing:
        *   For each block $i$: $X_{out} = \text{TimesBlock}_i(X_{in}, c_{cond})$
    4.  Inside `TimesBlock` (implied): The conditioning features are likely added or concatenated to the frequency domain representations or intermediate features within the block.

```python
# Code Snippet from bdc_TimesNet.py
label_emb = self.label_embedding(bd_labels)
label_cond = label_emb.unsqueeze(1) # [B, 1, d_model]
...
for i in range(self.layer):
    # Pass label conditioning to each TimesBlock
    enc_out = self.layer_norm(self.model[i](enc_out, label_cond))
```

---

## 4. TimesFM-based Generator (`bdc_timesFM.py`)

The `bdc_timesFM` model adapts Google's foundational TimesFM model (pretrained) for conditional trigger generation.

### Conditioning Mechanism
*   **Embedding**: `nn.Embedding(num_class, 64)`
*   **Injection Point**: Post-Backbone concatenation. The backbone is largely frozen (feature extractor), and conditioning happens at the readout stage.
*   **Process**:
    1.  Get base features: $H_{base} = \text{TimesFM\_Backbone}(X)$ (frozen/partially frozen)
    2.  Embed label: $c_{emb} = \text{Embedding}(y)$
    3.  **Concatenate**: $H_{combined} = \text{Concat}([H_{base}, \text{Broadcast}(c_{emb})], \text{dim}=-1)$
    4.  Trigger Head: Pass combined features through a learned MLP `trigger_head`.
        *   $\text{Trigger} = \text{MLP}(H_{combined})$

```python
# Code Snippet from bdc_timesFM.py
base_out = self.forecast(...) # Features from backbone
label_emb = self.label_embedding(bd_labels)
# Concatenate label conditioning with base output
conditioned_input = torch.cat([base_out, label_cond], dim=-1) 
trigger = self.trigger_head(conditioned_input)
```

---

## 5. Inverted Transformer (`bdc_inverted.py`)

The `bdc_inverted` model (iTransformer) uses a similar channel-independent approach to PatchTST but conditions on the variate embeddings.

### Conditioning Mechanism
*   **Embedding**: `nn.Embedding(num_class, d_model_bd)`
*   **Injection Point**: Added to the **variate tokens** (channel embeddings) before the attention mechanism.
*   **Process**:
    1.  Embed Variates: $H_{vars} = \text{DataEmbeddingInverted}(X)$ (Each variate becomes a token)
    2.  Embed Label: $c_{emb} = \text{Embedding}(y)$
    3.  **Inject**: Add label embedding to all variate tokens (broadcasting).
        *   $H_{input} = H_{vars} + \phi(c)$
    4.  Attention: Self-attention process across variates.

```python
# Code Snippet from bdc_inverted.py
enc_out = self.enc_embedding(x_enc, None) # [B, N, d_model]
label_cond = label_emb.unsqueeze(1) # [B, 1, d_model]
enc_out = enc_out + label_cond # Additive conditioning
enc_out, attns = self.encoder(enc_out)
```

---

## 6. Autoencoder CNN (`bdc_cnn_cae.py`)

The `bdc_CNN_CAE` model treats conditioning as an **additional input channel** (variate) in an autoencoder framework.

### Conditioning Mechanism
*   **Embedding**: `nn.Embedding(num_class, 64)`
*   **Injection Point**: Concatenated to the **input features** as an extra variate before the first convolution.
*   **Process**:
    1.  Embed label: $c_{emb} = \text{Embedding}(y)$
    2.  Project to Sequence Length:
        *   $c_{seq} = \text{Linear}(c_{emb})$ (shape $B \times L$)
    3.  Reshape as Channel:
        *   $c_{channel} = c_{seq}.\text{unsqueeze}(-1)$ (shape $B \times L \times 1$)
    4.  **Concatenate**:
        *   $X_{input} = [X_{enc}, c_{channel}]$ (shape $B \times L \times (D+1)$)
    5.  Encode/Decode: The network processes $D+1$ channels and decodes back to $D$ channels.

```python
# Code Snippet from bdc_cnn_cae.py
label_seq = self.label_proj(label_emb)  # [B, L]
label_channel = label_seq.unsqueeze(-1)  # [B, L, 1]
# Concatenate label channel with input
x_concat = torch.cat([x_enc, label_channel], dim=-1) # [B, L, D+1]
```

---

## General Implementation Notes

*   **Initialization**: If `bd_labels` is not provided during the forward pass, it defaults to class 0.
*   **Flexibility**: This design allows a single trained generator to produce $K$ distinct trigger patterns (where $K$ is the number of classes), simply by changing the input label vector.
