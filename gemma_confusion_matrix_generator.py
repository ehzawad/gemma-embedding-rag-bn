#!/usr/bin/env python3
"""
Comprehensive Confusion Matrix Generator for Bengali Legal RAG
============================================================
Generates detailed confusion matrix using EmbeddingGemma-300M model
with all production data and advanced evaluation metrics.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import logging
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    accuracy_score, 
    classification_report,
    precision_recall_fscore_support
)
from enhanced_gemma_rag import EnhancedEmbeddingGemmaRAG

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveConfusionMatrixGenerator:
    """Generate detailed confusion matrix and evaluation metrics"""
    
    def __init__(self, data_dir: str = "data/production_bengali_legal_dataset"):
        """Initialize with production dataset"""
        self.data_dir = Path(data_dir)
        self.rag_system = None
        self.all_questions = []
        self.all_labels = []
        self.category_stats = {}
        
    def load_all_data(self):
        """Load all production data for comprehensive evaluation"""
        logger.info("📂 Loading all production data...")
        
        csv_files = list(self.data_dir.glob("namjari_*.csv"))
        logger.info(f"Found {len(csv_files)} category files")
        
        for csv_file in csv_files:
            category = csv_file.stem.replace("namjari_", "")
            df = pd.read_csv(csv_file)
            
            questions = df['question'].tolist()
            # Clean and filter questions
            cleaned_questions = [q.strip() for q in questions if len(q.strip()) > 3]
            
            self.all_questions.extend(cleaned_questions)
            self.all_labels.extend([category] * len(cleaned_questions))
            
            self.category_stats[category] = {
                'total_samples': len(cleaned_questions),
                'file': csv_file.name
            }
        
        logger.info(f"✅ Loaded {len(self.all_questions)} total samples across {len(self.category_stats)} categories")
        
        # Print category distribution
        print("\n📊 Category Distribution:")
        print("-" * 50)
        for category, stats in sorted(self.category_stats.items()):
            print(f"{category:25} | {stats['total_samples']:4d} samples")
        print("-" * 50)
        print(f"{'TOTAL':25} | {len(self.all_questions):4d} samples")
    
    def initialize_rag_system(self, embedding_dim: int = 768):
        """Initialize RAG system with specified embedding dimension"""
        logger.info(f"🚀 Initializing Enhanced EmbeddingGemma RAG (dim={embedding_dim})...")
        self.rag_system = EnhancedEmbeddingGemmaRAG(
            data_dir=str(self.data_dir),
            confidence_threshold=0.5,
            embedding_dim=embedding_dim,
            use_task_prompts=True
        )
        logger.info("✅ RAG system initialized successfully")
    
    def evaluate_all_data(self) -> Dict:
        """Evaluate RAG system on all production data"""
        logger.info("🧪 Running comprehensive evaluation on all data...")
        
        predictions = []
        confidences = []
        query_times = []
        
        start_time = time.time()
        
        for i, (question, true_label) in enumerate(zip(self.all_questions, self.all_labels)):
            if i % 100 == 0:
                logger.info(f"Processing {i}/{len(self.all_questions)} samples...")
            
            result = self.rag_system.classify(question, top_k=5)
            predictions.append(result['predicted_tag'])
            confidences.append(result['confidence'])
            query_times.append(result.get('query_time', 0))
        
        total_time = time.time() - start_time
        
        # Calculate metrics
        accuracy = accuracy_score(self.all_labels, predictions)
        avg_confidence = np.mean(confidences)
        avg_query_time = np.mean(query_times)
        qps = len(self.all_questions) / total_time
        
        # Count confident predictions
        confident_count = sum(1 for c in confidences if c >= self.rag_system.confidence_threshold)
        confident_accuracy = accuracy_score(
            [true for true, conf in zip(self.all_labels, confidences) if conf >= self.rag_system.confidence_threshold],
            [pred for pred, conf in zip(predictions, confidences) if conf >= self.rag_system.confidence_threshold]
        ) if confident_count > 0 else 0
        
        return {
            'predictions': predictions,
            'true_labels': self.all_labels,
            'confidences': confidences,
            'accuracy': accuracy,
            'confident_accuracy': confident_accuracy,
            'average_confidence': avg_confidence,
            'confident_predictions': confident_count,
            'total_samples': len(self.all_questions),
            'average_query_time': avg_query_time,
            'queries_per_second': qps,
            'total_evaluation_time': total_time
        }
    
    def generate_confusion_matrix(self, eval_results: Dict, save_path: str = None) -> Tuple[np.ndarray, List[str]]:
        """Generate comprehensive confusion matrix with detailed visualization"""
        logger.info("📊 Generating comprehensive confusion matrix...")
        
        # Get unique labels
        all_categories = sorted(list(set(eval_results['true_labels'] + eval_results['predictions'])))
        
        # Create confusion matrix
        cm = confusion_matrix(eval_results['true_labels'], eval_results['predictions'], labels=all_categories)
        
        # Create detailed visualization
        plt.figure(figsize=(16, 12))
        
        # Clean category names for display
        display_labels = [cat.replace('namjari_', '').replace('_', ' ').title() for cat in all_categories]
        
        # Create heatmap
        sns.heatmap(cm, 
                   annot=True, 
                   fmt='d', 
                   cmap='Blues',
                   xticklabels=display_labels,
                   yticklabels=display_labels,
                   cbar_kws={'label': 'Number of Samples'})
        
        # Add comprehensive title with metrics
        title = (f'Bengali Legal RAG Confusion Matrix - EmbeddingGemma-300M\n'
                f'Overall Accuracy: {eval_results["accuracy"]:.1%} | '
                f'Confident Accuracy: {eval_results["confident_accuracy"]:.1%} | '
                f'Avg Confidence: {eval_results["average_confidence"]:.3f}\n'
                f'Speed: {eval_results["queries_per_second"]:.1f} QPS | '
                f'Samples: {eval_results["total_samples"]} | '
                f'Confident Predictions: {eval_results["confident_predictions"]}/{eval_results["total_samples"]}')
        
        plt.title(title, fontsize=12, pad=20)
        plt.xlabel('Predicted Category', fontsize=11)
        plt.ylabel('True Category', fontsize=11)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📊 Confusion matrix saved to: {save_path}")
        
        plt.show()
        
        return cm, all_categories
    
    def generate_detailed_report(self, eval_results: Dict) -> str:
        """Generate detailed classification report"""
        logger.info("📋 Generating detailed classification report...")
        
        # Generate classification report
        report = classification_report(
            eval_results['true_labels'], 
            eval_results['predictions'],
            target_names=[cat.replace('namjari_', '') for cat in sorted(set(eval_results['true_labels']))],
            digits=3
        )
        
        # Calculate per-category confidence statistics
        category_confidence = {}
        for true_label, pred_label, confidence in zip(eval_results['true_labels'], 
                                                     eval_results['predictions'], 
                                                     eval_results['confidences']):
            if true_label not in category_confidence:
                category_confidence[true_label] = []
            category_confidence[true_label].append(confidence)
        
        # Create comprehensive report
        detailed_report = f"""
Bengali Legal RAG System - Comprehensive Evaluation Report
=========================================================

SYSTEM CONFIGURATION:
- Model: EmbeddingGemma-300M (google/embeddinggemma-300m)
- FAISS Index: IndexFlatIP (Inner Product)
- Embedding Dimension: {self.rag_system.embedding_dim}D
- Task Prompts: {'Enabled' if self.rag_system.use_task_prompts else 'Disabled'}
- Confidence Threshold: {self.rag_system.confidence_threshold}

PERFORMANCE METRICS:
- Overall Accuracy: {eval_results['accuracy']:.1%}
- Confident Predictions Accuracy: {eval_results['confident_accuracy']:.1%}
- Average Confidence: {eval_results['average_confidence']:.3f}
- Confident Predictions: {eval_results['confident_predictions']}/{eval_results['total_samples']} ({eval_results['confident_predictions']/eval_results['total_samples']:.1%})

SPEED METRICS:
- Average Query Time: {eval_results['average_query_time']:.4f}s
- Queries Per Second: {eval_results['queries_per_second']:.1f} QPS
- Total Evaluation Time: {eval_results['total_evaluation_time']:.2f}s

DATASET STATISTICS:
- Total Samples: {eval_results['total_samples']}
- Categories: {len(set(eval_results['true_labels']))}
- Training Samples: {len(self.rag_system.train_questions)}
- Validation Samples: {len(self.rag_system.val_questions)}

PER-CATEGORY CONFIDENCE:
"""
        
        for category in sorted(category_confidence.keys()):
            confidences = category_confidence[category]
            avg_conf = np.mean(confidences)
            std_conf = np.std(confidences)
            detailed_report += f"- {category.replace('namjari_', ''):20}: {avg_conf:.3f} ± {std_conf:.3f}\n"
        
        detailed_report += f"\nCLASSIFICATION REPORT:\n{report}"
        
        return detailed_report
    
    def run_comprehensive_evaluation(self, save_dir: str = "confusion_matrix_results"):
        """Run comprehensive evaluation with 768D embeddings only"""
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        # Clean existing results
        for file in save_path.glob("*"):
            file.unlink()
        
        # Load all data first
        self.load_all_data()
        
        # Initialize RAG system with 768D embeddings
        logger.info(f"\nEvaluating with 768D embeddings...")
        self.initialize_rag_system(embedding_dim=768)
        
        # Run evaluation
        eval_results = self.evaluate_all_data()
        
        # Generate confusion matrix
        cm_path = save_path / "bengali_legal_confusion_matrix_768d.png"
        cm, categories = self.generate_confusion_matrix(eval_results, str(cm_path))
        
        # Generate detailed report
        report = self.generate_detailed_report(eval_results)
        
        # Save report
        report_path = save_path / "bengali_legal_evaluation_report_768d.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Detailed report saved to: {report_path}")
        
        # Print summary
        print(f"\nEVALUATION SUMMARY:")
        print(f"   Accuracy: {eval_results['accuracy']:.1%}")
        print(f"   Confident Accuracy: {eval_results['confident_accuracy']:.1%}")
        print(f"   Average Confidence: {eval_results['average_confidence']:.3f}")
        print(f"   Speed: {eval_results['queries_per_second']:.1f} QPS")
        print(f"   Confusion Matrix: {cm_path}")
        print(f"   Report: {report_path}")

def main():
    """Main execution function"""
    print("Bengali Legal RAG - Confusion Matrix Generator")
    print("=" * 50)
    
    # Initialize generator
    generator = ComprehensiveConfusionMatrixGenerator()
    
    # Run evaluation with 768D embeddings only
    generator.run_comprehensive_evaluation()
    
    print("\nEvaluation completed!")
    print("Check the 'confusion_matrix_results' directory for outputs")

if __name__ == "__main__":
    main()
