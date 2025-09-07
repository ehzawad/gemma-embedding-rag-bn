# Bengali Legal RAG System

Production-ready Bengali legal document RAG system with EmbeddingGemma-300M.

## Features

- EmbeddingGemma-300M with task-specific prompts and MRL optimization
- FAISS IndexFlatIP with automatic embedding regeneration
- Clean dataset: 1,418 samples across 14 namjari categories
- FastAPI server with comprehensive endpoints
- Zero data leakage with proper train/test separation

## Project Structure

```
gemma-embedding-rag-bn/
├── bengali_legal_rag.py              # Main RAG system (consolidated)
├── bengali_legal_api.py              # FastAPI server
├── train_vs_test_evaluator.py        # Train vs test evaluation
├── data/
│   ├── train/bengali_legal_train.csv # Training data (1,134 samples)
│   └── test/bengali_legal_test.csv   # Test data (284 samples)
├── faiss_cache/                      # FAISS index cache (auto-generated)
├── confusion_matrix_results/         # Evaluation results
└── requirements.txt                  # Dependencies
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the main system (auto-builds embeddings)
python bengali_legal_rag.py

# Generate train vs test confusion matrices
python train_vs_test_evaluator.py

# Start API server
python bengali_legal_api.py
```

## Dataset

- **Total Samples**: 1,418 Bengali legal questions
- **Categories**: 14 namjari-related legal document types
- **Training Split**: 1,134 samples (80%)
- **Test Split**: 284 samples (20%)
- **Split Method**: Stratified with zero overlap (data leakage eliminated)

## Technical Stack

- Model: google/embeddinggemma-300m (308M parameters)
- Vector Database: FAISS IndexFlatIP
- Embeddings: 768D with task prompts
- Framework: SentenceTransformers + FAISS + FastAPI

## API Usage

Start the API server:
```bash
source venv/bin/activate
python3 bengali_legal_api.py
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
  "confidence": 0.995,
  "predicted_tag": "fee",
  "is_relevant": true,
  "processing_time": 0.023
}
```
