# Related work

| Work | What it does | What we take / where we differ |
|---|---|---|
| **Large Concept Models** (Meta, 2024) | Predicts the next sentence embedding in a fixed space (SONAR) | Next-concept framing. We *learn* the concept space instead of fixing it |
| **Gist tokens** (Mu et al., 2023) | Compresses a prompt into a few learned tokens | Memory query token mechanism |
| **AutoCompressors** (Chevalier et al., 2023) | Recursively compresses segments into summary vectors | Rolling T0 compression |
| **ICAE: In-Context Autoencoder** (Ge et al., 2023) | LoRA encoder → memory slots → frozen LLM decoder; autoencoding and instruction pretraining | Closest template for Stages 1–2 |
| **Recurrent Memory Transformer** (Bulatov et al., 2022) | Memory tokens carried between segments | Carrying memory across chunks |
| **Titans** (Google, 2025) | Long-term memory updated at test time, alongside attention | T2 / persistence ideas |
| **Coconut** (Meta, 2024) | Reasoning with continuous thoughts instead of tokens | Latent reasoner |
| **Slot Attention** (Locatello et al., 2020) | Competitive assignment of inputs to object slots | T1 idea-slot formation |
| **Perceiver / Q-Former** | Fixed set of latents cross-attending to variable input | Alternative encoder design |
| **Byte Latent Transformer / H-Net** | Learned dynamic chunking | Possible future: variable-size concepts |
