# 📑 Hybrid ArXiv Classifier: BERT vs. LLM (LoRA)

**A production-grade, hybrid-cloud text classification system comparing Encoder (DistilBERT) and Generative (Llama-3) architectures.**

---

## 🚀 Overview

This project implements a robust document classification pipeline designed to categorize academic papers (ArXiv) into their respective fields (e.g., `cs.AI`, `math.ST`, `cs.CV`).

This project demonstrates a robust hybrid-cloud architecture and a modular 'Strategy Pattern' design.:
1.  **Hybrid Architecture:** Implements both a lightweight **Encoder** (DistilBERT) for speed and a **Generative Decoder** (Llama-3-8B) for reasoning.
2.  **Cloud-Native Inference:** Leverages **Modal** to decouple local orchestration from heavy GPU execution on the cloud.
3.  **Cost & Memory Optimization:** Uses **Sliding Window Chunking** for long documents and **LoRA (Low-Rank Adaptation) + 4-bit Quantization** to fine-tune an 8B model on consumer hardware.

---

## ⚡ Engineering Impact & Performance Analysis

This project solves the "Context vs. Compute" trade-off inherent in modern NLP pipelines. By architecting a hybrid system, we achieve the reliability of large models with the throughput of small ones.

### 1. 100x Throughput Improvement via Hybrid Routing
Instead of routing all traffic to a Generative LLM, this system uses a tiered architecture.
* **The Bottleneck:** Llama-3-8B inference takes **~1.5 seconds** per document.
* **The Solution:** DistilBERT handles standard classification in **~15 milliseconds**.
* **The Impact:** By reserving Llama-3 only for edge cases, the system theoretical throughput increases from **0.6 docs/sec** (Pure LLM) to **~60+ docs/sec** (Hybrid), depending on the routing ratio.

### 2. Solving the "Long Document" Limitation
Standard BERT models truncate text after 512 tokens, losing critical information in academic papers (Methods, Results, Conclusion).
* **Implementation:** Developed a **Sliding Window Inference Engine** that segments documents into 512-token chunks with a 25% overlap.
* **Aggregation:** Uses a majority-vote consensus algorithm to derive the final classification from multiple chunks, ensuring the *entire* document context is considered without exceeding memory limits.

### 3. Optimization for Consumer Hardware (QLoRA)
Deploying 8B parameter models typically requires enterprise A100 GPUs (80GB VRAM).
* **Optimization:** Implemented **4-bit Quantization** and **LoRA (Low-Rank Adaptation)**.
* **Result:** Compressed the active memory footprint from **~16GB** to **~5.5GB**, enabling the model to run on widely available **NVIDIA T4 or A10G** instances. This democratizes access to LLM reasoning without requiring heavy enterprise compute.

---

## 🏗️ System Architecture

The system follows a modular **Strategy Pattern**, allowing seamless switching between classification backends without altering the core application logic.

### 1. The "Specialist" Track (DistilBERT)
* **Goal:** High-throughput, low-latency classification.
* **Technique:** **Head Truncation** for training (focusing on abstracts) combined with **Sliding Window Inference** (aggregating votes across the full document).
* **Performance:** ~78% Accuracy. Extremely fast (ms/doc).

### 2. The "Generalist" Track (Llama-3 + LoRA)
* **Goal:** Nuanced understanding and zero-shot reasoning capabilities.
* **Technique:** **Instruction Tuning** using **Unsloth** (2x training speed). We model classification as a Causal Language Modeling task ("Next Token Prediction").
* **Optimization:** Fine-tuned using **QLoRA** (4-bit backbone + separate adapter layers) to fit an 8B model on a single A10G GPU.

---

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **Infrastructure:** [Modal](https://modal.com/) (Serverless GPU Cloud)
* **ML Frameworks:** PyTorch, Hugging Face Transformers, Unsloth (LLM Optimization)
* **Algorithms:** DistilBERT, Llama-3-8B, LoRA / QLoRA
* **Data Ops:** Hugging Face Datasets, Sliding Window Chunking
* **Dependency Management:** `uv`

---

## 📂 Project Structure

A clean separation of concerns ensures scalability and maintainability.

```text
src/
├── classifiers/       # Model Logic
│   ├── distilbert.py  ## Wrapper for Encoder model
│   └── llm_lora.py    ## Wrapper for Llama-3 + LoRA Adapters
├── processors/        # Data Manipulation
│   └── chunker.py     ## Sliding Window logic for handling 20k+ token docs
├── training/          # Training Scripts
│   ├── train_bert.py  ## Standard HF Trainer for DistilBERT
│   └── train_lora.py  ## Unsloth SFTTrainer for Llama
├── inference/         # Cloud Entry Points
│   ├── inference_distilbert.py
│   └── inference_lora.py
├── factory.py         # Strategy Pattern implementation
└── interfaces.py      # Abstract Base Classes (The Contract)
