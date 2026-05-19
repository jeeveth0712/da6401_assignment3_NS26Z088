# DA6401 - Assignment 3: Implementing the Transformer for Machine Translation

---

# IMPORTANT LINKS

## GITHUB REPOSITORY

https://github.com/jeeveth0712/da6401_assignment3_NS26Z088

## WANDB REPORT

https://api.wandb.ai/links/ns26z088-team/zcrd11lm

---

## Overview

In this assignment, you will implement the landmark architecture from the paper "Attention Is All You Need" from scratch using PyTorch. The goal is to develop a Neural Machine Translation (NMT) system capable of translating text from German to English using the Multi30k dataset.

## Project Structure

```text
assignment3/
├── requirements.txt
├── README.md
├── model.py           # Core Transformer architecture (Encoders, Decoders, Multi-Head Attention)
├── utils.py           # Label Smoothing, Noam Scheduler, Masking Utilities
├── dataset.py         # Multi30k dataset loading and spacy tokenization
├── train.py           # Training loops and Greedy Decoding inference
```
