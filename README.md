# Bengali Legal RAG System

Production-ready Bengali legal document RAG system with EmbeddingGemma-300M (Sept 2025 optimized).

## Features

- **Latest EmbeddingGemma-300M** with optimized prompt templates (Sept 2025)
- **94.7% Test Accuracy** with proper train/test separation
- **Real-time Performance Metrics** - no more hardcoded fake stats
- **Centralized Configuration** - single source of truth via config.py
- FAISS IndexFlatIP with automatic embedding regeneration
- Clean dataset: 1,418 samples across 14 namjari categories
- FastAPI server with comprehensive endpoints
- Zero data leakage with proper evaluation methodology

## Project Structure

```
gemma-embedding-rag-bn/
├── config.py                         # Centralized configuration management
├── bengali_legal_rag.py              # Main RAG system with Sept 2025 optimizations
├── bengali_legal_api.py              # FastAPI server with real-time metrics
├── train_vs_test_evaluator.py        # Test evaluation (no data leakage)
├── data/
│   ├── train/bengali_legal_train.csv # Training data (1,134 samples)
│   └── test/bengali_legal_test.csv   # Test data (284 samples)
├── faiss_cache/                      # FAISS index cache (auto-generated)
├── confusion_matrix_results/         # Evaluation results
├── CLAUDE.md                         # Developer guidance for Claude Code
└── requirements.txt                  # Dependencies
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the main system (auto-builds embeddings with optimized prompts)
/Users/ehz/venv-gemma-embedding/bin/python bengali_legal_rag.py

# Generate test evaluation (training evaluation removed due to data leakage)
/Users/ehz/venv-gemma-embedding/bin/python train_vs_test_evaluator.py

# Start API server with real-time metrics
/Users/ehz/venv-gemma-embedding/bin/python bengali_legal_api.py
```

## Dataset

- **Total Samples**: 1,418 Bengali legal questions
- **Categories**: 14 namjari-related legal document types
- **Training Split**: 1,134 samples (80%)
- **Test Split**: 284 samples (20%)
- **Split Method**: Stratified with zero overlap (data leakage eliminated)

## Technical Stack

- Model: google/embeddinggemma-300m (308M parameters, **Sept 2025 release**)
- **Latest Prompt Templates**: Optimized query/document prompts from Google AI
- Vector Database: FAISS IndexFlatIP  
- Embeddings: 768D with MRL support (512, 256, 128)
- Framework: SentenceTransformers + FAISS + FastAPI
- **Configuration**: Centralized via config.py (SystemConfig + EmbeddingGemmaPrompts)

## API Usage

Start the API server:
```bash
/Users/ehz/venv-gemma-embedding/bin/python bengali_legal_api.py
```

Server runs at: `http://127.0.0.1:8000`
API Documentation: `http://127.0.0.1:8000/docs`

### Main Chatbot Endpoint
```bash
curl -X POST "http://127.0.0.1:8000/land_em_bot/" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "আমি নামজারি করতে কত টাকা লাগবে?",
    "messages": "[]",
    "chat_id": "user123"
  }'
```

### Simple Classification
```bash
curl -X POST "http://127.0.0.1:8000/classify" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "নামজারির জন্য কি কি কাগজপত্র লাগে?"
  }'
```

### System Information
```bash
curl "http://127.0.0.1:8000/"           # Basic info
curl "http://127.0.0.1:8000/health"     # Health check
curl "http://127.0.0.1:8000/stats"      # Performance stats
curl "http://127.0.0.1:8000/tags"       # Available categories
```

### Response Format
```json
{
  "response": "আপনার প্রশ্নের উত্তর: fee সংক্রান্ত তথ্য...",
  "confidence": 0.832,
  "predicted_tag": "namjari_fee", 
  "is_relevant": true,
  "processing_time": 0.023
}
```

## Latest Improvements (Sept 2025)

### **🚀 EmbeddingGemma Optimizations**
- **Prompt Templates**: Using Google's latest optimized prompts
  - Query: `"task: search result | query: {content}"`
  - Document: `"title: {title} | text: {content}"`
- **94.7% Test Accuracy**: Improved from 94.0% with optimized prompts
- **Real-time Metrics**: Dynamic API responses, no more fake hardcoded stats

### **🏗️ Architecture Improvements** 
- **config.py**: Centralized configuration management
- **Performance Tracking**: Real-time QPS and accuracy monitoring  
- **Data Leakage Fix**: Training evaluation removed (was showing fake 100%)
- **MRL Support**: Matryoshka dimensions 768/512/256/128 ready
