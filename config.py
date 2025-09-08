#!/usr/bin/env python3
"""
Configuration Management for Bengali Legal RAG System
====================================================

Centralized configuration to eliminate duplicate constants and provide
single source of truth for all system parameters.
"""

import os
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass

# Environment setup (centralized)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

@dataclass
class SystemConfig:
    """Main system configuration class"""
    
    # Model Configuration
    MODEL_NAME: str = "google/embeddinggemma-300m"
    EMBEDDING_DIM: int = 768
    VALID_EMBEDDING_DIMS: List[int] = (768, 512, 256, 128)  # MRL supported dimensions
    USE_TASK_PROMPTS: bool = True
    TRUST_REMOTE_CODE: bool = True
    
    # Performance Configuration  
    BATCH_SIZE: int = 32
    NORMALIZE_EMBEDDINGS: bool = True
    SHOW_PROGRESS_BAR: bool = True
    CONVERT_TO_NUMPY: bool = True
    
    # RAG Configuration
    # CONFIDENCE_THRESHOLD: float = 0.75
    CONFIDENCE_THRESHOLD: float = 0.932
    TOP_K_SIMILAR: int = 5
    
    # File Paths
    TRAIN_FILE: str = "data/train/bengali_legal_train.csv"
    TEST_FILE: str = "data/test/bengali_legal_test.csv"
    
    # Cache Configuration
    CACHE_DIR: str = "faiss_cache"
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_DIR: str = "logs"
    IRRELEVANT_QUERIES_FILE: str = "logs/irrelevant_questions.csv"
    MAPPED_QUERIES_FILE: str = "logs/mapped_queries.csv"
    QUERY_LOG_FILE: str = "logs/query_log.txt"
    
    # API Configuration
    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8000
    API_TITLE: str = "Bengali Legal Bot API"
    API_DESCRIPTION: str = "EmbeddingGemma-powered Bengali legal intent detection"
    
    # Response Configuration
    RANDOM_RESPONSES: List[str] = (
        "আপনার প্রশ্নের জন্য ধন্যবাদ, দয়া করে আরও তথ্য জানতে আবার জিজ্ঞাসা করুন।",
        "আপনার প্রশ্নের জন্য ধন্যবাদ, দয়া করে আবার বলবেন।",
        "আপনার প্রশ্নটি পরিষ্কার নয়, দয়া করে আবার জিজ্ঞাসা করুন।",
        "প্রশ্নটি বোঝা যাচ্ছে না, আরও নির্দিষ্টভাবে জিজ্ঞাসা করলে ভালো হবে।",
        "আপনার প্রশ্নের জন্য ধন্যবাদ, বিস্তারিতভাবে পুনরায় জিজ্ঞাসা করবেন কি?",
        "প্রশ্নটি একটু পরিষ্কার করে আবার বলবেন।",
        "দয়া করে আরও বিস্তারিতভাবে আপনার প্রশ্নটি করবেন।",
        "আপনার প্রশ্নটি বুঝতে কিছুটা অসুবিধা হচ্ছে, দয়া করে আরেকবার জিজ্ঞাসা করুন।",
        "অনুগ্রহ করে প্রশ্নটি পুনরায় ব্যাখ্যা করবেন, আমি সাহায্য করতে প্রস্তুত।",
        "প্রশ্নটি স্পষ্ট নয়, দয়া করে আরেকবার জিজ্ঞাসা করুন।"
    )
    
    # Message History Configuration
    MAX_MESSAGE_HISTORY: int = 6
    RESET_THRESHOLD: int = 4

class EmbeddingGemmaPrompts:
    """
    Latest EmbeddingGemma prompt templates (Sept 2025)
    Based on official documentation from Google AI
    """
    
    # Task-specific prompt templates
    QUERY_TEMPLATE = "task: search result | query: {content}"
    DOCUMENT_TEMPLATE = "title: {title} | text: {content}" 
    DOCUMENT_NO_TITLE_TEMPLATE = "title: none | text: {content}"
    QUESTION_ANSWERING_TEMPLATE = "task: question answering | query: {content}"
    FACT_CHECKING_TEMPLATE = "task: fact checking | query: {content}"
    CLASSIFICATION_TEMPLATE = "task: classification | query: {content}"
    
    # Legacy fallback
    LEGACY_CLASSIFICATION = "Classification"
    
    @staticmethod
    def get_query_prompt(content: str) -> str:
        """Get optimized query prompt for search/retrieval"""
        return EmbeddingGemmaPrompts.QUERY_TEMPLATE.format(content=content)
    
    @staticmethod
    def get_document_prompt(content: str, title: Optional[str] = None) -> str:
        """Get optimized document prompt for indexing"""
        if title:
            return EmbeddingGemmaPrompts.DOCUMENT_TEMPLATE.format(title=title, content=content)
        return EmbeddingGemmaPrompts.DOCUMENT_NO_TITLE_TEMPLATE.format(content=content)
    
    @staticmethod
    def get_classification_prompt(content: str) -> str:
        """Get optimized classification prompt"""
        return EmbeddingGemmaPrompts.CLASSIFICATION_TEMPLATE.format(content=content)

# Global configuration instance
config = SystemConfig()

# Utility functions
def ensure_directories():
    """Ensure all required directories exist"""
    Path(config.CACHE_DIR).mkdir(exist_ok=True)
    Path(config.LOG_DIR).mkdir(exist_ok=True)

def get_cache_paths(embedding_dim: int) -> Dict[str, Path]:
    """Get cache file paths for given embedding dimension"""
    cache_dir = Path(config.CACHE_DIR)
    return {
        'index': cache_dir / f"rag_index_{embedding_dim}d.faiss",
        'metadata': cache_dir / f"rag_metadata_{embedding_dim}d.pkl"
    }

# Initialize directories on import
ensure_directories()
