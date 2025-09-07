# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Bengali Legal RAG System using EmbeddingGemma-300M for legal document intent classification. This is a production-ready system that processes Bengali legal questions and classifies them into 14 namjari-related categories using semantic embeddings and FAISS vector search.

## Commands

### Development Setup
```bash
pip install -r requirements.txt
```

### Running the System
```bash
# Run main RAG system (auto-builds embeddings and FAISS index)
/Users/ehz/venv-gemma-embedding/bin/python bengali_legal_rag.py

# Generate test evaluation (training evaluation is meaningless due to data leakage)
/Users/ehz/venv-gemma-embedding/bin/python train_vs_test_evaluator.py

# Start FastAPI server
/Users/ehz/venv-gemma-embedding/bin/python bengali_legal_api.py
```

### API Server
The FastAPI server runs at `http://127.0.0.1:8000` with interactive docs at `/docs`.

## Architecture

### Core Components

1. **bengali_legal_rag.py**: Main RAG system implementing `BengaliLegalRAG` class
   - EmbeddingGemma-300M model with task-specific prompts
   - FAISS IndexFlatIP for vector similarity search
   - Automatic embedding generation and caching
   - MRL (Matryoshka Representation Learning) optimization support
   - Train/test data separation with zero leakage

2. **bengali_legal_api.py**: FastAPI server with REST endpoints
   - Main chatbot endpoint: `/land_em_bot/`
   - Simple classification: `/classify`
   - System stats and health endpoints
   - Logging for irrelevant queries

3. **train_vs_test_evaluator.py**: Evaluation system
   - Generates confusion matrices for train and test splits
   - Performance analysis and visualization

### Data Structure
- `data/train/bengali_legal_train.csv`: Training data (1,134 samples, 80%)
- `data/test/bengali_legal_test.csv`: Test data (284 samples, 20%)  
- `faiss_cache/`: Auto-generated FAISS index storage
- `confusion_matrix_results/`: Evaluation outputs

### Key Classes

- `BengaliLegalRAG`: Main system class with methods:
  - `classify(question)`: Primary classification with confidence scoring
  - `load_data()`: Load and validate CSV data
  - `build_embeddings()`: Generate embeddings with task prompts
  - `build_faiss_index()`: Create FAISS vector index
  - `evaluate()`: Performance evaluation on test set

### Technical Details

- Model: google/embeddinggemma-300m (308M parameters)
- Embeddings: 768D with support for MRL dimensions (512, 256, 128)
- Vector DB: FAISS IndexFlatIP for inner product similarity
- Categories: 14 legal document types (fee, application, document, etc.)
- Confidence threshold: 0.5 (configurable)
- GPU support with automatic CPU fallback

### Important Evaluation Notes

**Data Leakage Warning**: Training set evaluation is meaningless in this RAG system because:
- FAISS index is built from training questions (`self.train_questions`)
- Evaluating on training data results in near-exact matches (100% accuracy)
- Only test set evaluation (94% accuracy) represents real performance
- The system does semantic search against the same training data it was built from

**Real Performance**: Use `train_vs_test_evaluator.py` which now only reports test accuracy.

### Environment Variables
- `TOKENIZERS_PARALLELISM=false`: Avoid tokenizer warnings
- `OMP_NUM_THREADS=1`: Control threading for consistency

### Python Environment
Use the specific virtual environment: `/Users/ehz/venv-gemma-embedding/bin/python`