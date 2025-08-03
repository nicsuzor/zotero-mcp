"""
ChromaDB client for semantic search functionality.

This module provides persistent vector database storage and embedding functions
for semantic search over Zotero libraries.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.config import Settings

logger = logging.getLogger(__name__)


class OpenAIEmbeddingFunction(EmbeddingFunction):
    """Custom OpenAI embedding function for ChromaDB."""
    
    def __init__(self, model_name: str = "text-embedding-3-small", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("openai package is required for OpenAI embeddings")
    
    def name(self) -> str:
        """Return the name of this embedding function."""
        return "openai"
    
    def __call__(self, input: Documents) -> Embeddings:
        """Generate embeddings using OpenAI API."""
        response = self.client.embeddings.create(
            model=self.model_name,
            input=input
        )
        return [data.embedding for data in response.data]


class GeminiEmbeddingFunction(EmbeddingFunction):
    """Custom Gemini embedding function for ChromaDB using google-genai."""
    
    def __init__(self, model_name: str = "models/text-embedding-004", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key is required")
        
        try:
            from google import genai
            from google.genai import types
            self.client = genai.Client(api_key=self.api_key)
            self.types = types
        except ImportError:
            raise ImportError("google-genai package is required for Gemini embeddings")
    
    def name(self) -> str:
        """Return the name of this embedding function."""
        return "gemini"
    
    def __call__(self, input: Documents) -> Embeddings:
        """Generate embeddings using Gemini API."""
        embeddings = []
        for text in input:
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=[text],
                config=self.types.EmbedContentConfig(
                    task_type="retrieval_document",
                    title="Zotero library document"
                )
            )
            embeddings.append(response.embeddings[0].values)
        return embeddings


class ChromaClient:
    """ChromaDB client for Zotero semantic search with multi-collection support."""
    
    def __init__(self, 
                 collection_name: str = "zotero_library",
                 persist_directory: Optional[str] = None,
                 embedding_model: str = "default",
                 embedding_config: Optional[Dict[str, Any]] = None):
        """
        Initialize ChromaDB client.
        
        Args:
            collection_name: Base name for ChromaDB collections
            persist_directory: Directory to persist the database
            embedding_model: Model to use for embeddings ('default', 'openai', 'gemini')
            embedding_config: Configuration for the embedding model
        """
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.embedding_config = embedding_config or {}
        
        # Set up persistent directory
        if persist_directory is None:
            # Use user's config directory by default
            config_dir = Path.home() / ".config" / "zotero-mcp"
            config_dir.mkdir(parents=True, exist_ok=True)
            persist_directory = str(config_dir / "chroma_db")
        
        self.persist_directory = persist_directory
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Set up embedding function
        self.embedding_function = self._create_embedding_function()
        
        # Initialize collections
        self.collections = {}
        self._init_collections()
    
    def _init_collections(self) -> None:
        """Initialize all required collections."""
        collection_configs = {
            "items": self.collection_name,  # Original metadata collection
            "fulltext": f"{self.collection_name}_fulltext",  # PDF text chunks
            "annotations": f"{self.collection_name}_annotations"  # PDF annotations
        }
        
        for collection_type, name in collection_configs.items():
            self.collections[collection_type] = self.client.get_or_create_collection(
                name=name,
                embedding_function=self.embedding_function
            )
        
        # Maintain backward compatibility
        self.collection = self.collections["items"]
    
    def _create_embedding_function(self) -> EmbeddingFunction:
        """Create the appropriate embedding function based on configuration."""
        if self.embedding_model == "openai":
            model_name = self.embedding_config.get("model_name", "text-embedding-3-small")
            api_key = self.embedding_config.get("api_key")
            return OpenAIEmbeddingFunction(model_name=model_name, api_key=api_key)
        
        elif self.embedding_model == "gemini":
            model_name = self.embedding_config.get("model_name", "models/text-embedding-004")
            api_key = self.embedding_config.get("api_key")
            return GeminiEmbeddingFunction(model_name=model_name, api_key=api_key)
        
        else:
            # Use ChromaDB's default embedding function (all-MiniLM-L6-v2)
            return chromadb.utils.embedding_functions.DefaultEmbeddingFunction()
    
    def add_documents(self, 
                     documents: List[str], 
                     metadatas: List[Dict[str, Any]], 
                     ids: List[str],
                     collection_type: str = "items") -> None:
        """
        Add documents to the specified collection.
        
        Args:
            documents: List of document texts to embed
            metadatas: List of metadata dictionaries for each document
            ids: List of unique IDs for each document
            collection_type: Type of collection ('items', 'fulltext', 'annotations')
        """
        try:
            collection = self.collections.get(collection_type, self.collection)
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Added {len(documents)} documents to {collection_type} collection")
        except Exception as e:
            logger.error(f"Error adding documents to {collection_type} collection: {e}")
            raise
    
    def upsert_documents(self,
                        documents: List[str],
                        metadatas: List[Dict[str, Any]],
                        ids: List[str],
                        collection_type: str = "items") -> None:
        """
        Upsert (update or insert) documents to the specified collection.
        
        Args:
            documents: List of document texts to embed
            metadatas: List of metadata dictionaries for each document
            ids: List of unique IDs for each document
            collection_type: Type of collection ('items', 'fulltext', 'annotations')
        """
        try:
            collection = self.collections.get(collection_type, self.collection)
            collection.upsert(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Upserted {len(documents)} documents to {collection_type} collection")
        except Exception as e:
            logger.error(f"Error upserting documents to {collection_type} collection: {e}")
            raise
    
    def search(self, 
               query_texts: List[str], 
               n_results: int = 10,
               where: Optional[Dict[str, Any]] = None,
               where_document: Optional[Dict[str, Any]] = None,
               collection_type: str = "items") -> Dict[str, Any]:
        """
        Search for similar documents in the specified collection.
        
        Args:
            query_texts: List of query texts
            n_results: Number of results to return
            where: Metadata filter conditions
            where_document: Document content filter conditions
            collection_type: Type of collection to search ('items', 'fulltext', 'annotations')
            
        Returns:
            Search results from ChromaDB
        """
        try:
            collection = self.collections.get(collection_type, self.collection)
            results = collection.query(
                query_texts=query_texts,
                n_results=n_results,
                where=where,
                where_document=where_document
            )
            logger.info(f"Semantic search in {collection_type} returned {len(results.get('ids', [[]])[0])} results")
            return results
        except Exception as e:
            logger.error(f"Error performing semantic search in {collection_type}: {e}")
            raise
    
    def search_all_collections(self,
                              query_texts: List[str],
                              n_results: int = 10,
                              where: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Search across all collections and return combined results.
        
        Args:
            query_texts: List of query texts
            n_results: Number of results to return per collection
            where: Metadata filter conditions
            
        Returns:
            Dictionary with results from each collection
        """
        results = {}
        
        for collection_type in self.collections.keys():
            try:
                collection_results = self.search(
                    query_texts=query_texts,
                    n_results=n_results,
                    where=where,
                    collection_type=collection_type
                )
                results[collection_type] = collection_results
            except Exception as e:
                logger.error(f"Error searching {collection_type} collection: {e}")
                results[collection_type] = {"ids": [[]], "distances": [[]], "documents": [[]], "metadatas": [[]]}
        
        return results
    
    def delete_documents(self, ids: List[str], collection_type: str = "items") -> None:
        """
        Delete documents from the specified collection.
        
        Args:
            ids: List of document IDs to delete
            collection_type: Type of collection ('items', 'fulltext', 'annotations')
        """
        try:
            collection = self.collections.get(collection_type, self.collection)
            collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents from {collection_type} collection")
        except Exception as e:
            logger.error(f"Error deleting documents from {collection_type} collection: {e}")
            raise
    
    def delete_item_from_all_collections(self, item_key: str) -> None:
        """
        Delete an item and all its related data from all collections.
        
        Args:
            item_key: Zotero item key
        """
        for collection_type in self.collections.keys():
            try:
                # For fulltext and annotations, we need to find and delete related documents
                if collection_type in ["fulltext", "annotations"]:
                    # Find documents related to this item
                    collection = self.collections[collection_type]
                    results = collection.get(where={"item_key": item_key})
                    if results["ids"]:
                        collection.delete(ids=results["ids"])
                        logger.info(f"Deleted {len(results['ids'])} {collection_type} documents for item {item_key}")
                else:
                    # For items collection, delete by item key directly
                    self.delete_documents([item_key], collection_type)
            except Exception as e:
                logger.error(f"Error deleting {item_key} from {collection_type} collection: {e}")
    
    def document_exists(self, doc_id: str, collection_type: str = "items") -> bool:
        """Check if a document exists in the specified collection."""
        try:
            collection = self.collections.get(collection_type, self.collection)
            result = collection.get(ids=[doc_id])
            return len(result['ids']) > 0
        except Exception:
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about all collections."""
        try:
            collection_info = {}
            total_count = 0
            
            for collection_type, collection in self.collections.items():
                try:
                    count = collection.count()
                    collection_info[collection_type] = {
                        "name": collection.name,
                        "count": count
                    }
                    total_count += count
                except Exception as e:
                    collection_info[collection_type] = {
                        "name": f"{self.collection_name}_{collection_type}",
                        "count": 0,
                        "error": str(e)
                    }
            
            return {
                "base_name": self.collection_name,
                "total_count": total_count,
                "embedding_model": self.embedding_model,
                "persist_directory": self.persist_directory,
                "collections": collection_info
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {
                "base_name": self.collection_name,
                "total_count": 0,
                "embedding_model": self.embedding_model,
                "persist_directory": self.persist_directory,
                "error": str(e)
            }
    
    def reset_collection(self, collection_type: Optional[str] = None) -> None:
        """Reset (clear) collections."""
        try:
            if collection_type:
                # Reset specific collection
                if collection_type in self.collections:
                    collection_name = self.collections[collection_type].name
                    self.client.delete_collection(name=collection_name)
                    self.collections[collection_type] = self.client.create_collection(
                        name=collection_name,
                        embedding_function=self.embedding_function
                    )
                    logger.info(f"Reset {collection_type} collection")
                    
                    # Update backward compatibility reference
                    if collection_type == "items":
                        self.collection = self.collections["items"]
            else:
                # Reset all collections
                for ctype in list(self.collections.keys()):
                    self.reset_collection(ctype)
                logger.info("Reset all collections")
        except Exception as e:
            logger.error(f"Error resetting collections: {e}")
            raise


def create_chroma_client(config_path: Optional[str] = None) -> ChromaClient:
    """
    Create a ChromaClient instance from configuration.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configured ChromaClient instance
    """
    # Default configuration
    config = {
        "collection_name": "zotero_library",
        "embedding_model": "default",
        "embedding_config": {}
    }
    
    # Load configuration from file if it exists
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                file_config = json.load(f)
                config.update(file_config.get("semantic_search", {}))
        except Exception as e:
            logger.warning(f"Error loading config from {config_path}: {e}")
    
    # Load configuration from environment variables
    env_embedding_model = os.getenv("ZOTERO_EMBEDDING_MODEL")
    if env_embedding_model:
        config["embedding_model"] = env_embedding_model
    
    # Set up embedding config from environment
    if config["embedding_model"] == "openai":
        openai_api_key = os.getenv("OPENAI_API_KEY")
        openai_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        if openai_api_key:
            config["embedding_config"] = {
                "api_key": openai_api_key,
                "model_name": openai_model
            }
    
    elif config["embedding_model"] == "gemini":
        gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        gemini_model = os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004")
        if gemini_api_key:
            config["embedding_config"] = {
                "api_key": gemini_api_key,
                "model_name": gemini_model
            }
    
    return ChromaClient(
        collection_name=config["collection_name"],
        embedding_model=config["embedding_model"],
        embedding_config=config["embedding_config"]
    )