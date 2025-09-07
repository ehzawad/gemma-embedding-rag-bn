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

- Overall Accuracy: 96.6%
- Average Confidence: 0.929
- Query Speed: 52.0 QPS
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

## API Endpoints

- `GET /` - System information
- `GET /health` - Health check
- `POST /classify` - Simple classification
- `POST /land_em_bot/` - Main chatbot endpoint
- `GET /stats` - System statistics
- `GET /tags` - Available categories
