#!/usr/bin/env python3
"""
Train vs Test Confusion Matrix Generator
=======================================
Generates separate confusion matrices for train and test data
"""

import os
import sys
import warnings
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

# Suppress all verbose output
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore")

# Suppress logging
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

sys.path.insert(0, str(Path(__file__).parent))
from bengali_legal_rag import BengaliLegalRAG

def evaluate_on_data(rag, questions, true_labels, data_type):
    """Evaluate RAG system on given data"""
    predictions = []
    
    for question in questions:
        result = rag.classify(question)
        predictions.append(result['predicted_tag'])
    
    accuracy = accuracy_score(true_labels, predictions)
    return {
        'predictions': predictions,
        'true_labels': true_labels,
        'accuracy': accuracy,
        'data_type': data_type
    }

def create_confusion_matrix(results, all_labels, title, filename):
    """Create and save confusion matrix"""
    cm = confusion_matrix(results['true_labels'], results['predictions'], labels=all_labels)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=all_labels, yticklabels=all_labels)
    plt.title(f'{title}\nAccuracy: {results["accuracy"]:.1%}', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def generate_test_evaluation():
    """Generate confusion matrix for test data only (training evaluation is meaningless due to data leakage)"""
    print("🎯 Generating test set evaluation...")
    print("⚠️  NOTE: Training set evaluation skipped - would show 100% accuracy due to data leakage")
    print("    (RAG system searches against same training questions used to build FAISS index)")
    
    rag = BengaliLegalRAG()
    
    # Evaluate ONLY on test data (real performance)
    print("📊 Evaluating on test data (real performance)...")
    test_results = evaluate_on_data(rag, rag.test_questions, rag.test_tags, "test")
    
    # Get all unique labels from test set
    all_labels = sorted(list(set(test_results['true_labels'] + test_results['predictions'])))
    
    # Create test confusion matrix
    test_path = "confusion_matrix_results/bengali_legal_test_confusion_matrix.png"
    create_confusion_matrix(
        test_results, all_labels,
        "Bengali Legal RAG - Test Set Performance (Real Accuracy)",
        test_path
    )
    
    # Create JSON report with test results only
    report_data = {
        "evaluation_summary": {
            "test_accuracy": test_results['accuracy'],
            "test_samples": len(test_results['true_labels']),
            "categories": len(all_labels),
            "note": "Training accuracy not reported due to data leakage (100% due to exact matching)"
        },
        "test_predictions": []
    }
    
    # Add test predictions
    for i, (question, true_label, predicted_label) in enumerate(zip(
        rag.test_questions, test_results['true_labels'], test_results['predictions']
    )):
        report_data["test_predictions"].append({
            "id": i + 1,
            "question": question,
            "actual_label": true_label,
            "predicted_label": predicted_label,
            "correct": true_label == predicted_label
        })
    
    # Save JSON report
    json_path = "confusion_matrix_results/test_predictions.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Test confusion matrix: {test_path}")
    print(f"📊 JSON predictions: {json_path}")
    print(f"🎯 Real test accuracy: {test_results['accuracy']:.1%}")
    print(f"📝 Training accuracy would be 100% (data leakage - not meaningful)")
    
    return test_results

if __name__ == "__main__":
    generate_test_evaluation()
