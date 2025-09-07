# Bengali Legal RAG System

Production-ready Bengali legal document RAG system with 96.6% accuracy.

## Features

- High accuracy: 96.6% on Bengali legal queries
- EmbeddingGemma-300M with task-specific prompts
- FAISS IndexFlatIP with CUDA acceleration (52 QPS)
- Production dataset: 1,485 samples across 14 categories
- FastAPI server with comprehensive endpoints

## Project Structure

```
gemma-embedding-rag-bn/
├── enhanced_gemma_rag.py             # Main RAG system
├── bengali_legal_api.py              # FastAPI server
├── gemma_confusion_matrix_generator.py # Evaluation
├── data/production_bengali_legal_dataset/ # Dataset + FAISS indices
├── confusion_matrix_results/         # Evaluation results
├── venv/                             # Virtual environment
└── requirements.txt                  # Dependencies
```

## Quick Start

```bash
# Activate virtual environment
source venv/bin/activate

# Run the main system
python3 enhanced_gemma_rag.py

# Start API server
python3 bengali_legal_api.py

# Generate confusion matrix
python3 gemma_confusion_matrix_generator.py
```

## Performance

- Overall Accuracy: 97.0%
- Average Confidence: 0.990
- Query Speed: 51.0 QPS
- Confident Predictions: 100% (1485/1485)

## Dataset

- Total Samples: 1,485 Bengali legal questions
- Categories: 14 legal document types (namjari-related)
- Training Split: 1,345 samples
- Validation Split: 140 samples

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
