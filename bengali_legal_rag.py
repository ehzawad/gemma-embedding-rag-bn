#!/usr/bin/env python3
"""
Bengali Legal RAG System with EmbeddingGemma
===========================================

Complete RAG system for Bengali legal intent classification using EmbeddingGemma-300M
with proper train/test separation and performance optimizations.
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
import tempfile
from sentence_transformers import SentenceTransformer
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Environment setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BengaliLegalRAG:
    """Complete Bengali Legal RAG System with EmbeddingGemma optimizations"""
    
    def __init__(self, 
                 train_file: str = "data/train/bengali_legal_train.csv",
                 test_file: str = "data/test/bengali_legal_test.csv", 
                 confidence_threshold: float = 0.5,
                 model_name: str = "google/embeddinggemma-300m",
                 embedding_dim: int = 768,
                 use_task_prompts: bool = True):
        """
        Initialize Bengali Legal RAG System
        
        Args:
            train_file: Path to training CSV file
            test_file: Path to test CSV file
            confidence_threshold: Minimum confidence for predictions
            model_name: EmbeddingGemma model name
            embedding_dim: Embedding dimension (768, 512, 256, or 128 for MRL)
            use_task_prompts: Whether to use task-specific prompts
        """
        logger.info(f"🚀 Initializing Bengali Legal RAG System (dim={embedding_dim})...")
        
        # Configuration
        self.train_file = Path(train_file)
        self.test_file = Path(test_file)
        self.confidence_threshold = confidence_threshold
        self.embedding_dim = embedding_dim
        self.use_task_prompts = use_task_prompts
        self.model_name = model_name
        
        # Initialize EmbeddingGemma model with optimizations
        logger.info(f"🔧 Loading EmbeddingGemma model (dim: {embedding_dim})...")
        
        # Suppress progress bars and verbose output
        import logging
        transformers_logger = logging.getLogger("transformers")
        transformers_logger.setLevel(logging.ERROR)
        
        # Suppress sentence-transformers progress bars
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # Disable tqdm progress bars
        import warnings
        warnings.filterwarnings("ignore")
        
        # Disable tqdm globally
        try:
            import tqdm
            tqdm.tqdm.disable = True
        except ImportError:
            pass
        
        self.model = SentenceTransformer(
            self.model_name, 
            trust_remote_code=True,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Validate embedding dimension
        valid_dims = [768, 512, 256, 128]
        if embedding_dim not in valid_dims:
            logger.warning(f"Embedding dim {embedding_dim} not in {valid_dims}, using 768")
            self.embedding_dim = 768
        
        # Performance optimization
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        # Data storage
        self.train_questions: List[str] = []
        self.train_tags: List[str] = []
        self.test_questions: List[str] = []
        self.test_tags: List[str] = []
        self.index: Optional[faiss.Index] = None
        
        # Cache paths - store in project directory
        cache_dir = Path(__file__).parent / "faiss_cache"
        cache_dir.mkdir(exist_ok=True)
        self.index_path = cache_dir / f"rag_index_{embedding_dim}d.faiss"
        self.metadata_path = cache_dir / f"rag_metadata_{embedding_dim}d.pkl"
        
        # Initialize system
        self._load_data()
        self._load_or_build_index()
        
        logger.info("✅ Bengali Legal RAG System ready!")
    
    def _get_prompt_name(self) -> Optional[str]:
        """Get EmbeddingGemma prompt name for classification task"""
        return "Classification" if self.use_task_prompts else None
    
    def _load_data(self):
        """Load training and test data"""
        logger.info("📂 Loading training and test data...")
        
        # Load training data
        train_df = pd.read_csv(self.train_file)
        self.train_questions = train_df['question'].tolist()
        self.train_tags = train_df['tag'].tolist()
        
        # Load test data
        test_df = pd.read_csv(self.test_file)
        self.test_questions = test_df['question'].tolist()
        self.test_tags = test_df['tag'].tolist()
        
        logger.info(f"✅ Loaded {len(self.train_questions)} train, {len(self.test_questions)} test samples")
    
    def _load_or_build_index(self):
        """Load existing FAISS index or build new one"""
        if self.index_path.exists() and self.metadata_path.exists():
            logger.info(f"📂 Loading cached FAISS index...")
            try:
                self.index = faiss.read_index(str(self.index_path))
                
                with open(self.metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                
                # Verify metadata matches
                if (metadata.get('embedding_dim') == self.embedding_dim and 
                    metadata.get('use_task_prompts') == self.use_task_prompts and
                    metadata.get('num_samples') == len(self.train_questions)):
                    
                    logger.info(f"✅ Loaded cached index: {self.index.ntotal} vectors")
                    return
                else:
                    logger.info("⚠️ Cache mismatch, rebuilding...")
            except Exception as e:
                logger.info(f"⚠️ Cache load failed: {e}, rebuilding...")
        
        # Build new index
        self._build_index()
        
        # Save index and metadata
        logger.info(f"💾 Saving FAISS index...")
        faiss.write_index(self.index, str(self.index_path))
        
        metadata = {
            'embedding_dim': self.embedding_dim,
            'use_task_prompts': self.use_task_prompts,
            'num_samples': len(self.train_questions),
            'created_at': time.time()
        }
        
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
    
    def _build_index(self):
        """Build FAISS index from training data"""
        logger.info("🔨 Building FAISS index...")
        
        prompt_name = self._get_prompt_name()
        
        # Generate embeddings
        if prompt_name:
            embeddings = self.model.encode(
                self.train_questions, 
                prompt_name=prompt_name,
                batch_size=32, 
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
        else:
            embeddings = self.model.encode(
                self.train_questions, 
                batch_size=32, 
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
        
        # Build FAISS index
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings.astype('float32'))
        
        logger.info(f"✅ FAISS index built: {self.index.ntotal} vectors, {embeddings.shape[1]}D")
    
    def classify(self, query: str, top_k: int = 5) -> Dict:
        """
        Classify Bengali legal query using semantic search
        
        Args:
            query: Bengali text to classify
            top_k: Number of similar examples to consider
            
        Returns:
            Classification result with confidence score
        """
        if self.index is None:
            raise RuntimeError("Index not built. System not properly initialized.")
        
        # Generate query embedding
        prompt_name = self._get_prompt_name()
        
        if prompt_name:
            query_embedding = self.model.encode(
                [query], 
                prompt_name=prompt_name,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
        else:
            query_embedding = self.model.encode(
                [query], 
                convert_to_numpy=True,
                normalize_embeddings=True
            )
        
        # Search similar questions
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Process results
        similar_questions = []
        for i in range(top_k):
            similar_questions.append({
                'question': self.train_questions[indices[0][i]],
                'tag': self.train_tags[indices[0][i]],
                'similarity': float(scores[0][i])
            })
        
        # Primary prediction from most similar
        predicted_tag = similar_questions[0]['tag']
        confidence = similar_questions[0]['similarity']
        is_confident = confidence >= self.confidence_threshold
        
        return {
            'query': query,
            'predicted_tag': predicted_tag,
            'confidence': confidence,
            'is_confident': is_confident,
            'similar_questions': similar_questions
        }
    
    def evaluate(self, use_test_set: bool = True) -> Dict:
        """
        Evaluate system performance
        
        Args:
            use_test_set: If True, use test set; otherwise use training set
            
        Returns:
            Evaluation metrics
            
        Note:
            Training set evaluation will show inflated accuracy (~100%) due to data leakage.
            The FAISS index is built from training questions, so evaluating on the same
            training questions results in near-exact matches. Use test set for real performance.
        """
        if use_test_set:
            questions = self.test_questions
            true_tags = self.test_tags
            logger.info("🧪 Evaluating on test set...")
        else:
            questions = self.train_questions
            true_tags = self.train_tags
            logger.warning("⚠️  Evaluating on training set - expect inflated accuracy due to data leakage")
        
        predictions = []
        confidences = []
        
        for question, true_tag in zip(questions, true_tags):
            result = self.classify(question)
            predictions.append(result['predicted_tag'])
            confidences.append(result['confidence'])
        
        # Calculate metrics
        accuracy = accuracy_score(true_tags, predictions)
        avg_confidence = np.mean(confidences)
        
        # Confident predictions accuracy
        confident_mask = np.array(confidences) >= self.confidence_threshold
        if confident_mask.sum() > 0:
            confident_accuracy = accuracy_score(
                np.array(true_tags)[confident_mask],
                np.array(predictions)[confident_mask]
            )
        else:
            confident_accuracy = 0.0
        
        return {
            'accuracy': accuracy,
            'confident_accuracy': confident_accuracy,
            'average_confidence': avg_confidence,
            'confident_predictions': confident_mask.sum(),
            'total_samples': len(questions),
            'predictions': predictions,
            'true_labels': true_tags,
            'confidences': confidences
        }
    
    def create_confusion_matrix(self, save_path: str = None, use_test_set: bool = True) -> Tuple[np.ndarray, List[str]]:
        """Create and save confusion matrix"""
        logger.info("📊 Generating confusion matrix...")
        
        # Get evaluation results
        eval_results = self.evaluate(use_test_set=use_test_set)
        
        # Get unique labels
        all_labels = sorted(list(set(eval_results['true_labels'] + eval_results['predictions'])))
        
        # Create confusion matrix
        cm = confusion_matrix(eval_results['true_labels'], eval_results['predictions'], labels=all_labels)
        
        # Create visualization
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=[label.replace('namjari_', '') for label in all_labels],
                   yticklabels=[label.replace('namjari_', '') for label in all_labels])
        
        dataset_type = "Test Set" if use_test_set else "Training Set"
        plt.title(f'Bengali Legal RAG Confusion Matrix - {dataset_type}\n'
                 f'Accuracy: {eval_results["accuracy"]:.1%} | '
                 f'Confident Accuracy: {eval_results["confident_accuracy"]:.1%} | '
                 f'Avg Confidence: {eval_results["average_confidence"]:.3f}')
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
            'test_samples': len(self.test_questions),
            'total_tags': len(set(self.train_tags)),
            'embedding_dim': self.embedding_dim,
            'confidence_threshold': self.confidence_threshold,
            'device': str(self.model.device)
        }

def main():
    """Demo of the Bengali Legal RAG system"""
    print("🚀 Bengali Legal RAG System")
    print("=" * 50)
    
    # Initialize system
    rag = BengaliLegalRAG()
    
    # Test sample queries
    test_queries = [
        "আমি নামজারি করতে কত টাকা লাগবে?",
        "নামজারির জন্য কি কি কাগজপত্র লাগে?",
        "আমি নামজারি আবেদন কিভাবে করবো?"
    ]
    
    print(f"\n📝 Sample Classifications:")
    for query in test_queries:
        result = rag.classify(query)
        status = "✅" if result['is_confident'] else "⚠️"
        print(f"{status} {query}")
        print(f"   → {result['predicted_tag'].replace('namjari_', '')} (confidence: {result['confidence']:.3f})")
    
    # Show system stats
    stats = rag.get_stats()
    print(f"\n📊 System Statistics:")
    print(f"  Model: {stats['model']}")
    print(f"  Training Samples: {stats['training_samples']}")
    print(f"  Test Samples: {stats['test_samples']}")
    print(f"  Categories: {stats['total_tags']}")
    print(f"  Embedding Dim: {stats['embedding_dim']}")
    print(f"  Threshold: {stats['confidence_threshold']}")
    
    # Evaluate on test set
    eval_results = rag.evaluate(use_test_set=True)
    print(f"\n🎯 Test Set Results:")
    print(f"  Accuracy: {eval_results['accuracy']:.1%}")
    print(f"  Confident Accuracy: {eval_results['confident_accuracy']:.1%}")
    print(f"  Average Confidence: {eval_results['average_confidence']:.3f}")
    print(f"  Confident Predictions: {eval_results['confident_predictions']}/{eval_results['total_samples']}")
    
    # Note: Confusion matrix generation moved to train_vs_test_evaluator.py
    print(f"\n📊 To generate confusion matrix, run: python train_vs_test_evaluator.py")
    
    print(f"\n✅ System ready!")
    return rag

if __name__ == "__main__":
    main()
