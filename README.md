# Transformer Architectures for Historical Speech Analysis

## Overview
This repository contains a custom, from-scratch implementation of Transformer models built in PyTorch. The project explores both bidirectional (Encoder) and autoregressive (Decoder) architectures, focusing on optimizing attention mechanisms to reduce computational overhead.

The project is divided into two primary tasks:
1. **Speech Authorship Classification:** An Encoder-based model jointly trained from scratch with a feedforward classifier. It predicts which of three American politicians delivered a given historical speech segment.
2. **Autoregressive Language Modeling:** A GPT-like, word-level Decoder-based model exploring different attention paradigms, specifically evaluating standard attention against sparse attention patterns.

## Architectural Highlights & Custom Implementations

### 1. Sparse Attention (Local Window Attention)
To reduce computational overhead, I implemented a custom `WindowedTransformerDecoder` utilizing local window attention. By restricting the attention matrix to a localized window (Window Size = 5), the model deliberately sacrifices a global receptive field and O(N^2) contextual awareness. However, this sparse attention pattern successfully reduced memory overhead while improving the baseline language model perplexity.

### 2. End-to-End Joint Training
The classification pipeline features a `TransformerEncoder` that processes input sentences into high-dimensional, context-rich embeddings. These embeddings are averaged across the sequence dimension. They are then fed into a custom feedforward classifier to make predictions about which politician spoke the segment. Both components are trained simultaneously.

## Results
* **Classification:** Achieved an **85.47% test accuracy** on the 3-way political speech classification task.
* **Language Modeling:** The implementation of Local Window Attention improved the model's perplexity by **10%** (reducing from 169.6 to 152.8) compared to the standard global attention baseline.

## Repository Structure
* `transformer.py`: Core architectures including `MultiHeadAttention`, `TransformerEncoder`, `TransformerDecoder`, and the custom `WindowedTransformerDecoder`.
* `main.py`: Training loops, evaluation metrics, and hyperparameter configurations.
* `dataset.py` & `tokenizer.py`: Data loading and a simple word-level tokenizer.
* `utilities.py`: Helper functions for sanity checking the attention implementation. It checks if the rows sum to 1 and visualizes the attention matrix.