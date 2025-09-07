#!/usr/bin/env python3
"""
Bengali Legal RAG System with EmbeddingGemma
===========================================

Complete RAG system for Bengali legal intent classification using EmbeddingGemma-300M
with proper train/test separation and latest optimization features (Sept 2025).
"""

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

# Import centralized configuration
from config import config, EmbeddingGemmaPrompts, get_cache_paths

logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)

class BengaliLegalRAG:
    """Complete Bengali Legal RAG System with EmbeddingGemma optimizations"""
    
    def __init__(self, 
                 train_file: Optional[str] = None,
                 test_file: Optional[str] = None, 
                 confidence_threshold: Optional[float] = None,
                 model_name: Optional[str] = None,
                 embedding_dim: Optional[int] = None,
                 use_task_prompts: Optional[bool] = None):
        """
        Initialize Bengali Legal RAG System
        
        Args:
            train_file: Path to training CSV file (default from config)
            test_file: Path to test CSV file (default from config)
            confidence_threshold: Minimum confidence for predictions (default from config)
            model_name: EmbeddingGemma model name (default from config)
            embedding_dim: Embedding dimension - supports MRL: 768, 512, 256, 128 (default from config)
            use_task_prompts: Whether to use latest EmbeddingGemma prompt templates (default from config)
        """
        # Use centralized config with optional overrides
        self.train_file = Path(train_file or config.TRAIN_FILE)
        self.test_file = Path(test_file or config.TEST_FILE)
        self.confidence_threshold = confidence_threshold or config.CONFIDENCE_THRESHOLD
        self.embedding_dim = embedding_dim or config.EMBEDDING_DIM
        self.use_task_prompts = use_task_prompts if use_task_prompts is not None else config.USE_TASK_PROMPTS
        self.model_name = model_name or config.MODEL_NAME
        
        logger.info(f"🚀 Initializing Bengali Legal RAG System (dim={self.embedding_dim})...")
        
        # Initialize EmbeddingGemma model with optimizations
        logger.info(f"🔧 Loading EmbeddingGemma model (dim: {self.embedding_dim})...")
        
        # Suppress progress bars and verbose output
        transformers_logger = logging.getLogger("transformers")
        transformers_logger.setLevel(logging.ERROR)
        
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
            trust_remote_code=config.TRUST_REMOTE_CODE,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Validate embedding dimension (MRL support)
        if self.embedding_dim not in config.VALID_EMBEDDING_DIMS:
            logger.warning(f"Embedding dim {self.embedding_dim} not in {config.VALID_EMBEDDING_DIMS}, using 768")
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
        
        # Cache paths from centralized config
        cache_paths = get_cache_paths(self.embedding_dim)
        self.index_path = cache_paths['index']
        self.metadata_path = cache_paths['metadata']
        
        # Initialize system
        self._load_data()
        self._load_or_build_index()
        
        logger.info("✅ Bengali Legal RAG System ready!")
    
    def _get_query_prompt(self, content: str) -> Optional[str]:
        """Get optimized EmbeddingGemma prompt for query embedding (Sept 2025)"""
        if self.use_task_prompts:
            return EmbeddingGemmaPrompts.get_query_prompt(content)
        return content
    
    def _get_document_prompt(self, content: str, title: Optional[str] = None) -> Optional[str]:
        """Get optimized EmbeddingGemma prompt for document embedding (Sept 2025)"""
        if self.use_task_prompts:
            return EmbeddingGemmaPrompts.get_document_prompt(content, title)
        return content
    
    def _get_legacy_prompt_name(self) -> Optional[str]:
        """Legacy prompt method for backward compatibility"""
        return EmbeddingGemmaPrompts.LEGACY_CLASSIFICATION if self.use_task_prompts else None
    
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
        """Build FAISS index from training data using latest EmbeddingGemma optimizations"""
        logger.info("🔨 Building FAISS index with optimized prompts...")
        
        # Prepare training documents with optimized prompts
        if self.use_task_prompts:
            logger.info("📝 Using latest EmbeddingGemma document prompts (Sept 2025)")
            prompted_questions = [
                self._get_document_prompt(question, title=None) 
                for question in self.train_questions
            ]
        else:
            prompted_questions = self.train_questions
        
        # Generate embeddings with config parameters
        embeddings = self.model.encode(
            prompted_questions, 
            batch_size=config.BATCH_SIZE,
            show_progress_bar=config.SHOW_PROGRESS_BAR,
            convert_to_numpy=config.CONVERT_TO_NUMPY,
            normalize_embeddings=config.NORMALIZE_EMBEDDINGS
        )
        
        # Build FAISS index
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings.astype('float32'))
        
        logger.info(f"✅ FAISS index built: {self.index.ntotal} vectors, {embeddings.shape[1]}D")
    
    def classify(self, query: str, top_k: Optional[int] = None) -> Dict:
        """
        Classify Bengali legal query using semantic search with optimized EmbeddingGemma prompts
        
        Args:
            query: Bengali text to classify
            top_k: Number of similar examples to consider (default from config)
            
        Returns:
            Classification result with confidence score
        """
        if self.index is None:
            raise RuntimeError("Index not built. System not properly initialized.")
            
        # Use config default if not specified
        top_k = top_k or config.TOP_K_SIMILAR
        
        # Generate query embedding with optimized prompts
        if self.use_task_prompts:
            prompted_query = self._get_query_prompt(query)
            logger.debug(f"🔍 Using optimized query prompt: {prompted_query[:50]}...")
        else:
            prompted_query = query
            
        query_embedding = self.model.encode(
            [prompted_query],
            convert_to_numpy=config.CONVERT_TO_NUMPY,
            normalize_embeddings=config.NORMALIZE_EMBEDDINGS
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
