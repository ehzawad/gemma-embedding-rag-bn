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

def generate_train_test_evaluation():
    """Generate separate confusion matrices for train and test data"""
    print("🎯 Generating train vs test evaluation...")
    
    rag = BengaliLegalRAG()
    
    # Evaluate on train data (should perform better)
    print("📊 Evaluating on train data...")
    train_results = evaluate_on_data(rag, rag.train_questions, rag.train_tags, "train")
    
    # Evaluate on test data (real performance)
    print("📊 Evaluating on test data...")
    test_results = evaluate_on_data(rag, rag.test_questions, rag.test_tags, "test")
    
    # Get all unique labels
    all_labels = sorted(list(set(
        train_results['true_labels'] + train_results['predictions'] +
        test_results['true_labels'] + test_results['predictions']
    )))
    
    # Create train confusion matrix
    train_path = "confusion_matrix_results/train_confusion_matrix.png"
    create_confusion_matrix(
        train_results, all_labels, 
        "Bengali Legal RAG - Train Data Performance", 
        train_path
    )
    
    # Create test confusion matrix
    test_path = "confusion_matrix_results/test_confusion_matrix.png"
    create_confusion_matrix(
        test_results, all_labels,
        "Bengali Legal RAG - Test Data Performance",
        test_path
    )
    
    # Create JSON report with both results
    report_data = {
        "evaluation_summary": {
            "train_accuracy": train_results['accuracy'],
            "test_accuracy": test_results['accuracy'],
            "train_samples": len(train_results['true_labels']),
            "test_samples": len(test_results['true_labels']),
            "categories": len(all_labels),
            "performance_difference": train_results['accuracy'] - test_results['accuracy']
        },
        "train_predictions": [],
        "test_predictions": []
    }
    
    # Add train predictions
    for i, (question, true_label, predicted_label) in enumerate(zip(
        rag.train_questions, train_results['true_labels'], train_results['predictions']
    )):
        report_data["train_predictions"].append({
            "id": i + 1,
            "question": question,
            "actual_label": true_label,
            "predicted_label": predicted_label,
            "correct": true_label == predicted_label
        })
    
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
    json_path = "confusion_matrix_results/train_test_predictions.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Train confusion matrix: {train_path}")
    print(f"✅ Test confusion matrix: {test_path}")
    print(f"📊 JSON predictions: {json_path}")
    print(f"🎯 Train accuracy: {train_results['accuracy']:.1%}")
    print(f"🎯 Test accuracy: {test_results['accuracy']:.1%}")
    print(f"📈 Difference: {(train_results['accuracy'] - test_results['accuracy'])*100:+.1f}%")
    
    return train_results, test_results

if __name__ == "__main__":
    generate_train_test_evaluation()
