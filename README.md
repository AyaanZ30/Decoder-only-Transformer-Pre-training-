# Decoder-only-Transformer-Pre-training-
Overview

This repository contains a custom implementation of a decoder-only transformer neural network, pre-trained from scratch on a corpus of Shakespearean text, including monologues and dialogues. Unlike large language models (LLMs) such as GPT, which are fine-tuned and optimized in multiple stages (like Proximal Policy Optimization (PPO) for text generation), this model focuses solely on pre-training, making it a minimalistic and foundational implementation of the transformer architecture.
Foundation in Transformer Architecture

The model is based on the seminal "Attention is All You Need" paper, which introduced the world to the transformer architecture. Transformers revolutionized natural language processing (NLP) by replacing traditional recurrent networks with self-attention mechanisms. This model strictly follows the core principles of the original transformer design, employing only the decoder component and focusing on attention-based sequence generation.
Minimalistic Approach

While modern LLMs like GPT undergo various stages of development, including:

    Pre-training on vast and diverse datasets,
    Fine-tuning for task-specific objectives,
    Reinforcement learning from human feedback (RLHF) to align responses with human preferences (often utilizing PPO),

this model remains deliberately minimal. It is designed as a pure pre-training model without additional fine-tuning or reinforcement learning phases. The goal here is to demonstrate the power of transformers in generating text based solely on pre-training on Shakespearean data.
Future Directions

In the future, this model could be extended to include more advanced techniques such as fine-tuning on modern datasets or even integrating reinforcement learning for further optimization.
