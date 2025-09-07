#!/usr/bin/env python3
"""
Enhanced Bengali Legal RAG with Latest EmbeddingGemma Optimizations
=================================================================
Implements Matryoshka Representation Learning, task prompts, and performance optimizations
based on the latest EmbeddingGemma improvements.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
import time
import torch
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Environment setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedEmbeddingGemmaRAG:
    """Enhanced Bengali Legal RAG with latest EmbeddingGemma optimizations"""
    
    def __init__(self, 
                 data_dir: str = "data/production_bengali_legal_dataset", 
                 confidence_threshold: float = 0.5,
                 embedding_dim: int = 768,
                 use_task_prompts: bool = True):
        """
        Initialize with latest EmbeddingGemma optimizations
        
        Args:
            data_dir: Directory containing training data
            confidence_threshold: Minimum confidence for predictions
            embedding_dim: Embedding dimension (768, 512, 256, or 128 for MRL)
            use_task_prompts: Whether to use task-specific prompts
        """
        logger.info(f"🚀 Loading Enhanced EmbeddingGemma-300M (dim={embedding_dim})...")
        
        self.model = SentenceTransformer("google/embeddinggemma-300m")
        self.confidence_threshold = confidence_threshold
        self.data_dir = Path(data_dir)
        self.embedding_dim = embedding_dim
        self.use_task_prompts = use_task_prompts
        
        # Validate embedding dimension for MRL
        valid_dims = [768, 512, 256, 128]
        if embedding_dim not in valid_dims:
            logger.warning(f"Embedding dim {embedding_dim} not in {valid_dims}, using 768")
            self.embedding_dim = 768
        
        # Performance optimization
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        # Data storage
        self.train_questions = []
        self.train_tags = []
        self.val_questions = []
        self.val_tags = []
        self.index = None
        
        # Index cache paths
        self.index_path = self.data_dir / f"gemma_index_{embedding_dim}d.faiss"
        self.metadata_path = self.data_dir / f"gemma_metadata_{embedding_dim}d.pkl"
        
        # Load and build index
        self._load_data()
        self._load_or_build_index()
    
    def _apply_task_prompt(self, text: str, task_type: str = "query") -> str:
        """Apply task-specific prompts as recommended by EmbeddingGemma"""
        if not self.use_task_prompts:
            return text
            
        if task_type == "query":
            return f"query: {text}"
        elif task_type == "document":
            return f"document: {text}"
        elif task_type == "search":
            return f"search query: {text}"
        else:
            return text
    
    def _load_data(self):
        """Load training and validation data with enhanced processing"""
        logger.info("Loading and preprocessing data...")
        
        csv_files = list(self.data_dir.glob("namjari_*.csv"))
        
        for csv_file in csv_files:
            tag = csv_file.stem.replace("namjari_", "")
            df = pd.read_csv(csv_file)
            questions = df['question'].tolist()
            
            # Enhanced preprocessing
            processed_questions = []
            for question in questions:
                # Clean and normalize text
                cleaned = question.strip()
                if len(cleaned) > 3:  # Filter very short questions
                    processed_questions.append(cleaned)
            
            # Split into train/validation
            val_questions = processed_questions[-10:]
            train_questions = processed_questions[:-10]
            
            self.val_questions.extend(val_questions)
            self.val_tags.extend([tag] * len(val_questions))
            
            self.train_questions.extend(train_questions)
            self.train_tags.extend([tag] * len(train_questions))
        
        logger.info(f"✅ Loaded {len(self.train_questions)} training and {len(self.val_questions)} validation samples")
    
    def _load_or_build_index(self):
        """Load existing index or build new one with caching"""
        if self.index_path.exists() and self.metadata_path.exists():
            logger.info(f"📂 Loading cached FAISS index from {self.index_path}")
            try:
                # Load index
                self.index = faiss.read_index(str(self.index_path))
                
                # Load metadata
                with open(self.metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                
                # Verify metadata matches current setup
                if (metadata.get('embedding_dim') == self.embedding_dim and 
                    metadata.get('use_task_prompts') == self.use_task_prompts and
                    metadata.get('num_samples') == len(self.train_questions)):
                    
                    logger.info(f"✅ Loaded cached index: {self.index.ntotal} vectors, {self.index.d} dimensions")
                    return
                else:
                    logger.info("⚠️ Cached index metadata mismatch, rebuilding...")
            except Exception as e:
                logger.info(f"⚠️ Failed to load cached index: {e}, rebuilding...")
        
        # Build new index
        self._build_enhanced_index()
        
        # Save index and metadata
        logger.info(f"💾 Saving FAISS index to {self.index_path}")
        faiss.write_index(self.index, str(self.index_path))
        
        metadata = {
            'embedding_dim': self.embedding_dim,
            'use_task_prompts': self.use_task_prompts,
            'num_samples': len(self.train_questions),
            'model_name': 'google/embeddinggemma-300m',
            'created_at': time.time()
        }
        
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"✅ Index and metadata saved!")

    def _build_enhanced_index(self):
        """Build enhanced FAISS index with MRL and optimizations"""
        logger.info("Building enhanced FAISS index with latest optimizations...")
        
        # Apply task prompts to training documents
        prompted_questions = [
            self._apply_task_prompt(q, "document") for q in self.train_questions
        ]
        
        # Generate embeddings
        start_time = time.time()
        embeddings = self.model.encode(
            prompted_questions, 
            batch_size=32, 
            show_progress_bar=True,
            convert_to_numpy=True
        )
        embedding_time = time.time() - start_time
        
        # Apply Matryoshka Representation Learning (MRL) - truncate to desired dimension
        if self.embedding_dim < 768:
            logger.info(f"Applying MRL: truncating embeddings to {self.embedding_dim} dimensions")
            embeddings = embeddings[:, :self.embedding_dim]
        
        # Normalize for inner product search (recommended for EmbeddingGemma)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Build optimized FAISS index
        start_time = time.time()
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings.astype('float32'))
        index_time = time.time() - start_time
        
        logger.info(f"✅ Enhanced FAISS IndexFlatIP ready:")
        logger.info(f"   - Vectors: {self.index.ntotal}")
        logger.info(f"   - Dimensions: {embeddings.shape[1]}")
        logger.info(f"   - Embedding time: {embedding_time:.2f}s")
        logger.info(f"   - Index build time: {index_time:.2f}s")
    
    def query(self, question: str, k: int = 3) -> Dict:
        """Enhanced query with task prompts and optimizations"""
        # Apply task prompt to query
        prompted_query = self._apply_task_prompt(question, "query")
        
        # Generate embedding
        start_time = time.time()
        query_embedding = self.model.encode(
            [prompted_query], 
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        # Apply MRL if needed
        if self.embedding_dim < 768:
            query_embedding = query_embedding[:, :self.embedding_dim]
        
        # Normalize
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        query_time = time.time() - start_time
        
        # Process results
        results = []
        for i in range(k):
            results.append({
                'question': self.train_questions[indices[0][i]],
                'category': self.train_tags[indices[0][i]],
                'confidence': float(scores[0][i])
            })
        
        primary_prediction = results[0]['category'] if results[0]['confidence'] >= self.confidence_threshold else 'uncertain'
        
        return {
            'query': question,
            'prediction': primary_prediction,
            'confidence': results[0]['confidence'],
            'query_time': query_time,
            'similar_questions': results
        }
    
    def classify(self, query: str, top_k: int = 5) -> Dict:
        """
        Classify Bengali legal query using semantic search
        
        Args:
            query: Bengali text to classify
            top_k: Number of similar examples to consider
            
        Returns:
            Classification result with confidence score
        """
        result = self.query(query, k=top_k)
        
        return {
            'query': query,
            'predicted_tag': result['prediction'],
            'confidence': result['confidence'],
            'is_confident': result['confidence'] >= self.confidence_threshold,
            'similar_questions': result['similar_questions']
        }
    
    def evaluate_validation_set(self) -> Dict:
        """Evaluate system on validation set and return metrics"""
        logger.info("🧪 Evaluating on validation set...")
        
        predictions = []
        true_labels = []
        confidences = []
        
        for question, true_tag in zip(self.val_questions, self.val_tags):
            result = self.classify(question)
            predictions.append(result['predicted_tag'])
            true_labels.append(true_tag)
            confidences.append(result['confidence'])
        
        # Calculate accuracy
        accuracy = accuracy_score(true_labels, predictions)
        avg_confidence = np.mean(confidences)
        
        # Count confident predictions
        confident_predictions = sum(1 for c in confidences if c >= self.confidence_threshold)
        
        return {
            'accuracy': accuracy,
            'average_confidence': avg_confidence,
            'confident_predictions': confident_predictions,
            'total_samples': len(self.val_questions),
            'predictions': predictions,
            'true_labels': true_labels,
            'confidences': confidences
        }
    
    def create_confusion_matrix(self, save_path: str = None) -> Tuple[np.ndarray, List[str]]:
        """Create and save confusion matrix"""
        logger.info("📊 Generating confusion matrix...")
        
        # Get evaluation results
        eval_results = self.evaluate_validation_set()
        
        # Get unique labels
        all_labels = sorted(list(set(eval_results['true_labels'] + eval_results['predictions'])))
        
        # Create confusion matrix
        cm = confusion_matrix(eval_results['true_labels'], eval_results['predictions'], labels=all_labels)
        
        # Create visualization
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=[label.replace('namjari_', '') for label in all_labels],
                   yticklabels=[label.replace('namjari_', '') for label in all_labels])
        
        plt.title(f'Bengali Legal RAG Confusion Matrix\nAccuracy: {eval_results["accuracy"]:.1%} | Confidence: {eval_results["average_confidence"]:.3f}')
        plt.xlabel('Predicted Category')
        plt.ylabel('True Category')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📊 Confusion matrix saved to: {save_path}")
        
        plt.show()
        
        return cm, all_labels

    def get_stats(self) -> Dict:
        """Get system statistics"""
        return {
            'model': 'google/embeddinggemma-300m',
            'faiss_index': 'IndexFlatIP',
            'training_samples': len(self.train_questions),
            'validation_samples': len(self.val_questions),
            'total_tags': len(set(self.train_tags)),
            'confidence_threshold': self.confidence_threshold,
            'device': str(self.model.device)
        }

def main():
    """Main demo function"""
    print("🚀 Enhanced Bengali Legal RAG System")
    print("=" * 50)
    print("📋 EmbeddingGemma-300M + FAISS IndexFlatIP")
    print("=" * 50)
    
    # Initialize system
    rag = EnhancedEmbeddingGemmaRAG()
    
    # Test sample queries  
    print(f"\n📝 Testing Sample Queries:")
    print("-" * 40)
    
    test_queries = [
        "আমি নামজারি করতে কত টাকা লাগবে?",
        "নামজারির জন্য কি কি কাগজপত্র লাগে?",
        "আমি নামজারি আবেদন কিভাবে করবো?"
    ]
    
    for query in test_queries:
        result = rag.classify(query)
        status = "✅" if result['is_confident'] else "⚠️"
        
        print(f"{status} {query}")
        print(f"   → {result['predicted_tag'].replace('namjari_', '')} (confidence: {result['confidence']:.3f})")
    
    # Show system stats
    stats = rag.get_stats()
    print(f"\n📊 System Statistics:")
    print(f"  Model: {stats['model']}")
    print(f"  FAISS Index: {stats['faiss_index']}")
    print(f"  Training Samples: {stats['training_samples']}")
    print(f"  Validation Samples: {stats['validation_samples']}")
    print(f"  Tags: {stats['total_tags']}")
    print(f"  Threshold: {stats['confidence_threshold']}")
    
    print(f"\n✅ System ready!")
    
    return rag

def main():
    """Main demo function"""
    print("🚀 Enhanced Bengali Legal RAG System")
    print("=" * 50)
    print("📋 EmbeddingGemma-300M + FAISS IndexFlatIP")
    print("=" * 50)
    
    # Initialize system
    rag = EnhancedEmbeddingGemmaRAG()
    
    # Test sample queries  
    print(f"\n📝 Testing Sample Queries:")
    print("-" * 40)
    
    test_queries = [
        "আমি নামজারি করতে কত টাকা লাগবে?",
        "নামজারির জন্য কি কি কাগজপত্র লাগে?",
        "আমি নামজারি আবেদন কিভাবে করবো?"
    ]
    
    for query in test_queries:
        result = rag.classify(query)
        status = "✅" if result['is_confident'] else "⚠️"
        
        print(f"{status} {query}")
        print(f"   → {result['predicted_tag'].replace('namjari_', '')} (confidence: {result['confidence']:.3f})")
    
    # Show system stats
    stats = rag.get_stats()
    print(f"\n📊 System Statistics:")
    print(f"  Model: {stats['model']}")
    print(f"  FAISS Index: {stats['faiss_index']}")
    print(f"  Training Samples: {stats['training_samples']}")
    print(f"  Validation Samples: {stats['validation_samples']}")
    print(f"  Tags: {stats['total_tags']}")
    print(f"  Threshold: {stats['confidence_threshold']}")
    
    # Generate confusion matrix
    print(f"\n🎨 Generating confusion matrix...")
    cm_path = "bengali_legal_confusion_matrix.png"
    cm, labels = rag.create_confusion_matrix(save_path=cm_path)
    
    # Evaluate system
    eval_results = rag.evaluate_validation_set()
    print(f"\n🎯 VALIDATION RESULTS:")
    print(f"   📊 Accuracy: {eval_results['accuracy']:.1%}")
    print(f"   💎 Confidence: {eval_results['average_confidence']:.3f}")
    print(f"   📈 Confident Predictions: {eval_results['confident_predictions']}/{eval_results['total_samples']}")
    
    print(f"\n✅ System ready!")
    print(f"📊 Confusion matrix saved to: {cm_path}")
    
    return rag

if __name__ == "__main__":
    main()
