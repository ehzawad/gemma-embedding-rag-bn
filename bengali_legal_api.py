#!/usr/bin/env python3
"""
Bengali Legal API with EmbeddingGemma RAG
========================================

FastAPI endpoint for Bengali legal intent detection using EmbeddingGemma-300M RAG system.
Based on user's reference code structure but optimized for EmbeddingGemma.
"""

import time
import json
import random
import csv
from typing import List, Dict, Any
from pathlib import Path
import sys

from fastapi import FastAPI
from pydantic import BaseModel
from filelock import FileLock
import logging

# Import centralized configuration and RAG system
from config import config
from bengali_legal_rag import BengaliLegalRAG

# Logging setup
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)

# Initialize FastAPI app with config
app = FastAPI(title=config.API_TITLE, description=config.API_DESCRIPTION)

# RAG system initialization with centralized config
logger.info("Initializing Bengali Legal RAG system...")
rag = BengaliLegalRAG()  # Uses config defaults
logger.info("✅ RAG system ready!")

# Pydantic models
class RequestBody(BaseModel):
    question: str
    messages: str
    chat_id: str

class Question(BaseModel):
    question: str

# Performance tracking for dynamic metrics
class PerformanceTracker:
    """Track real-time performance metrics"""
    def __init__(self):
        self.request_times = []
        self.request_count = 0
    
    def record_request(self, processing_time: float):
        self.request_times.append(processing_time)
        self.request_count += 1
        # Keep only last 1000 requests for rolling metrics
        if len(self.request_times) > 1000:
            self.request_times.pop(0)
    
    def get_qps(self) -> float:
        """Calculate queries per second"""
        if len(self.request_times) < 2:
            return 0.0
        return min(1.0 / (sum(self.request_times[-10:]) / len(self.request_times[-10:])), 100.0)

# Global performance tracker
perf_tracker = PerformanceTracker()

# Get real-time system metrics
def get_dynamic_metrics() -> Dict[str, Any]:
    """Calculate real-time system performance metrics"""
    # Get actual test evaluation
    try:
        test_eval = rag.evaluate(use_test_set=True)
        return {
            "test_accuracy": f"{test_eval['accuracy']:.1%}",
            # High confidence accuracy removed 
            "average_confidence": f"{test_eval['average_confidence']:.3f}",
            "query_speed": f"{perf_tracker.get_qps():.1f} QPS"
        }
    except Exception as e:
        logger.warning(f"Failed to get dynamic metrics: {e}")
        return {
            "test_accuracy": "N/A",
            # High confidence accuracy removed
            "average_confidence": "N/A", 
            "query_speed": f"{perf_tracker.get_qps():.1f} QPS"
        }

def log_irrelevant_query(question: str, filepath: str = None):
    """Log queries with low confidence scores"""
    filepath = filepath or config.IRRELEVANT_QUERIES_FILE
    lock_path = filepath + ".lock"
    with FileLock(lock_path):
        if Path(filepath).exists():
            try:
                import pandas as pd
                df = pd.read_csv(filepath)
                if question not in df["question"].values:
                    df = pd.concat([df, pd.DataFrame([{"question": question}])], ignore_index=True)
                    df.to_csv(filepath, index=False, encoding='utf-8')
            except Exception:
                # Fallback to CSV writing
                with open(filepath, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([question])
        else:
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["question"])
                writer.writerow([question])

def log_mapped_query(user_input: str, matched_question: str, tag: str, confidence: float, 
                     filepath: str = None):
    """Log successful query mappings"""
    filepath = filepath or config.MAPPED_QUERIES_FILE
    lock_path = filepath + ".lock"
    with FileLock(lock_path):
        file_exists = Path(filepath).exists()
        
        with open(filepath, mode='a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["user_input", "matched_question", "tag", "confidence"])
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow({
                "user_input": user_input,
                "matched_question": matched_question,
                "tag": tag,
                "confidence": confidence
            })

@app.get("/")
def read_root():
    """API root with real-time performance metrics"""
    dynamic_metrics = get_dynamic_metrics()
    return {
        "message": "Bengali Legal Bot API with EmbeddingGemma (Sept 2025 optimized)",
        "model": config.MODEL_NAME,
        **dynamic_metrics
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_ready": rag.index is not None,
        "training_samples": len(rag.train_questions),
        "model_device": str(rag.model.device)
    }

@app.post("/classify")
async def classify_query(question: Question):
    """Simple classification endpoint with performance tracking"""
    try:
        start_time = time.time()
        
        result = rag.classify(question.question)
        
        processing_time = time.time() - start_time
        
        # Record performance metrics
        perf_tracker.record_request(processing_time)
        
        return {
            "query": question.question,
            "predicted_tag": result['predicted_tag'],
            "confidence": result['confidence'],
            "meets_threshold": result['meets_threshold'],
            "similar_questions": result['similar_questions'][:3],  # Top 3
            "processing_time": processing_time,
            "threshold": config.CONFIDENCE_THRESHOLD
        }
        
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        return {"error": str(e)}

@app.post("/land_em_bot/")
async def get_response(body: RequestBody):
    """
    Main chatbot endpoint following user's reference structure
    Adapted for EmbeddingGemma RAG system
    """
    try:
        start_time = time.time()
        
        # Extract request data
        user_actual_input = body.question
        messages = body.messages
        chat_id = body.chat_id
        
        logger.info(f"Processing query: {user_actual_input}")
        
        # Parse messages
        try:
            messages = json.loads(messages) if isinstance(messages, str) else messages
        except:
            messages = []
        
        # Add user input to messages
        messages.append({"role": "user", "content": user_actual_input})
        
        # Get intent detection result
        intent_result = rag.classify(user_actual_input)
        
        predicted_tag = intent_result['predicted_tag']
        confidence = intent_result['confidence']
        meets_threshold = intent_result['meets_threshold']
        similar_questions = intent_result['similar_questions']
        
        # Determine response based on confidence
        if confidence >= config.CONFIDENCE_THRESHOLD and meets_threshold:
            # High confidence - use the most similar question as response
            response = f"আপনার প্রশ্নের উত্তর: {predicted_tag.replace('namjari_', '')} সংক্রান্ত তথ্য। " \
                      f"সবচেয়ে মিল: {similar_questions[0]['question']}"
            response_tag = predicted_tag
            is_relevant = True
            
            # Log successful mapping
            log_mapped_query(
                user_input=user_actual_input,
                matched_question=similar_questions[0]['question'],
                tag=predicted_tag,
                confidence=confidence
            )
            
        else:
            # Low confidence - use fallback response
            response = random.choice(config.RANDOM_RESPONSES)
            response_tag = "out_of_scope"
            is_relevant = False
            
            # Log irrelevant query
            log_irrelevant_query(user_actual_input)
        
        # Add response extension for relevant queries
        response_extension = " আপনার কি আর কোন প্রশ্ন আছে?"
        
        # Don't add extension for specific tags
        if response_tag not in ['greetings', 'goodbye', 'repeat_again', 'agent_calling']:
            response = response + response_extension
        
        # Add assistant response to messages
        messages.append({
            "role": "assistant", 
            "content": response, 
            "tag": response_tag
        })
        
        # Handle conversation flags (simplified)
        is_repeat_again = response_tag == "repeat_again"
        is_conversation_finished = response_tag == "goodbye"
        is_agent_calling = response_tag == "agent_calling"
        
        # Limit message history using config
        if len(messages) >= config.MAX_MESSAGE_HISTORY:
            messages = messages[-config.RESET_THRESHOLD:]
            logger.info("Message history reset!")
        
        # Convert messages back to string
        messages_str = json.dumps(messages, ensure_ascii=False)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Record performance metrics
        perf_tracker.record_request(processing_time)
        
        # Detailed logging
        log_string = f"""
        Query: {user_actual_input}
        Predicted Tag: {predicted_tag}
        Confidence: {confidence:.3f}
        Meets Threshold: {meets_threshold}
        Processing Time: {processing_time:.3f}s
        Similar Question: {similar_questions[0]['question'] if similar_questions else 'N/A'}
        Response: {response[:100]}...
        """
        
        logger.info(log_string)
        
        # Save detailed log
        with open(config.QUERY_LOG_FILE, "a", encoding="utf-8") as log_file:
            log_file.write(log_string + "\n" + "-" * 80 + "\n")
        
        return {
            "id": 0,
            "response": response,
            "response_tag": response_tag,
            "time_taken": processing_time,
            "messages": messages_str,
            "is_repeat_again": is_repeat_again,
            "is_conversation_finished": is_conversation_finished,
            "is_agent_calling": is_agent_calling,
            "probability": confidence,
            "user_input": f"User: {user_actual_input}\nSimilar: {similar_questions[0]['question'] if similar_questions else 'N/A'}",
            "is_relevant": is_relevant,
            "confidence": confidence,
            "predicted_tag": predicted_tag
        }
        
    except Exception as e:
        logger.error(f"API request failed: {e}")
        return {"error": str(e)}

@app.get("/stats")
def get_stats():
    """Get system statistics with real-time metrics"""
    stats = rag.get_stats()
    dynamic_metrics = get_dynamic_metrics()
    return {
        "model": stats['model'],
        "faiss_index": stats['faiss_index'], 
        "training_samples": stats['training_samples'],
        "test_samples": stats['test_samples'],
        "total_tags": stats['total_tags'],
        "confidence_threshold": stats['confidence_threshold'],
        "device": stats['device'],
        **dynamic_metrics
    }

@app.get("/tags")
def get_available_tags():
    """Get list of available tags"""
    unique_tags = sorted(list(set(rag.train_tags)))
    return {
        "tags": unique_tags,
        "count": len(set(rag.train_tags))
    }

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Bengali Legal API with EmbeddingGemma (Sept 2025 optimized)")
    print("=" * 60)
    print(f"🚀 Training Samples: {len(rag.train_questions)}")
    print(f"📊 Test Samples: {len(rag.test_questions)}")
    print(f"🏷️  Total Tags: {len(set(rag.train_tags))}")
    print(f"🎯 Confidence Threshold: {rag.confidence_threshold}")
    print(f"💻 Device: {rag.model.device}")
    print(f"📝 Using Latest EmbeddingGemma Prompts: {rag.use_task_prompts}")
    print("=" * 60)
    print(f"API will be available at: http://{config.API_HOST}:{config.API_PORT}")
    print(f"Docs available at: http://{config.API_HOST}:{config.API_PORT}/docs")
    
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)
