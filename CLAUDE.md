# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Bengali Legal RAG System using EmbeddingGemma-300M (Sept 2025 optimized) for legal document intent classification. Production-ready system with 94.7% test accuracy that processes Bengali legal questions and classifies them into 14 namjari-related categories using latest optimized prompts and semantic embeddings.

## Commands

### Development Setup
```bash
pip install -r requirements.txt
```

### Running the System
```bash
# Run main RAG system (auto-builds embeddings with optimized prompts)
/Users/ehz/venv-gemma-embedding/bin/python bengali_legal_rag.py

# Generate test evaluation only (training removed due to data leakage)
/Users/ehz/venv-gemma-embedding/bin/python train_vs_test_evaluator.py

# Start FastAPI server with real-time metrics
/Users/ehz/venv-gemma-embedding/bin/python bengali_legal_api.py
```

### API Server
The FastAPI server runs at `http://127.0.0.1:8000` with interactive docs at `/docs`.

## Architecture

### Core Components

1. **config.py**: Centralized configuration management (NEW)
   - `SystemConfig` class with all parameters
   - `EmbeddingGemmaPrompts` with latest Sept 2025 templates
   - Single source of truth for all settings

2. **bengali_legal_rag.py**: Main RAG system implementing `BengaliLegalRAG` class
   - EmbeddingGemma-300M with latest optimized prompts
   - FAISS IndexFlatIP for vector similarity search
   - Automatic embedding generation and caching
   - MRL (Matryoshka Representation Learning) support: 768/512/256/128
   - Train/test data separation with zero leakage

3. **bengali_legal_api.py**: FastAPI server with real-time metrics
   - Main chatbot endpoint: `/land_em_bot/`
   - Simple classification: `/classify`
   - Real-time performance tracking with `PerformanceTracker`
   - Dynamic metrics instead of hardcoded values
   - System stats and health endpoints

4. **train_vs_test_evaluator.py**: Test-only evaluation system
   - Generates confusion matrix for test data only
   - Training evaluation removed (was showing fake 100% due to data leakage)
   - Performance analysis and visualization

### Data Structure
- `data/train/bengali_legal_train.csv`: Training data (1,134 samples, 80%)
- `data/test/bengali_legal_test.csv`: Test data (284 samples, 20%)  
- `faiss_cache/`: Auto-generated FAISS index storage
- `confusion_matrix_results/`: Evaluation outputs

### Key Classes

- `SystemConfig`: Centralized configuration (config.py)
  - All model parameters, file paths, thresholds
  - Environment setup and validation

- `EmbeddingGemmaPrompts`: Latest prompt templates (config.py)
  - `get_query_prompt()`: "task: search result | query: {content}"
  - `get_document_prompt()`: "title: {title} | text: {content}"
  - `get_classification_prompt()`: For Bengali legal classification

- `BengaliLegalRAG`: Main system class with methods:
  - `classify(question)`: Primary classification with optimized prompts
  - `_load_data()`: Load and validate CSV data
  - `_build_index()`: Generate embeddings with latest prompts
  - `evaluate()`: Performance evaluation (warns about training data leakage)

- `PerformanceTracker`: Real-time metrics (bengali_legal_api.py)
  - `record_request()`: Track processing times
  - `get_qps()`: Calculate queries per second

### Technical Details

- Model: google/embeddinggemma-300m (308M parameters, **Sept 2025 release**)
- **Latest Prompt Optimization**: Query/document prompts from Google AI documentation
- Embeddings: 768D with MRL support for dimensions (512, 256, 128)
- Vector DB: FAISS IndexFlatIP for inner product similarity
- Categories: 14 legal document types (fee, application, document, etc.)
- **Real Performance**: 94.7% test accuracy (improved from 94.0%)
- Configuration: Centralized via config.py
- GPU support with automatic CPU fallback

### Important Evaluation Notes

**Data Leakage Fixed**: 
- Training set evaluation removed (was showing fake 100% accuracy)
- FAISS index built from training questions → evaluating on same data = cheating
- Only test set evaluation (94.7%) represents real performance
- API now shows dynamic metrics instead of hardcoded fake values

**Configuration Management**:
- All settings centralized in `config.py`
- `SystemConfig` class eliminates duplicate constants
- `EmbeddingGemmaPrompts` provides latest templates
- Import `from config import config` to access settings

### Environment Variables
- `TOKENIZERS_PARALLELISM=false`: Avoid tokenizer warnings (set in config.py)
- `OMP_NUM_THREADS=1`: Control threading for consistency

### Python Environment
Use the specific virtual environment: `/Users/ehz/venv-gemma-embedding/bin/python`

### Latest Improvements (Sept 2025)
- **EmbeddingGemma optimized prompts**: 0.7% accuracy gain
- **Real-time API metrics**: No more hardcoded fake stats
- **Centralized configuration**: Single source of truth
- **Data leakage elimination**: Honest evaluation methodology