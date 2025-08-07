"""
Semantic search functionality for Zotero MCP.

This module provides semantic search capabilities by integrating ChromaDB
with the existing Zotero client to enable vector-based similarity search
over research libraries.
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
import semchunk
from pyzotero import zotero

from zotero_mcp.server import format_item_metadata, get_item_fulltext

from .chroma_client import ChromaClient, create_chroma_client
from .client import convert_to_markdown, get_attachment_details, get_zotero_client
from .utils import format_creators

logger = logging.getLogger(__name__)


class ZoteroSemanticSearch:
    """Semantic search interface for Zotero libraries using ChromaDB."""
    
    def __init__(self, 
                 chroma_client: Optional[ChromaClient] = None,
                 config_path: Optional[str] = None):
        """
        Initialize semantic search.
        
        Args:
            chroma_client: Optional ChromaClient instance
            config_path: Path to configuration file
        """
        self.chroma_client = chroma_client or create_chroma_client(config_path)
        self.zotero_client = get_zotero_client()
        self.config_path = config_path
        
        # Load update configuration
        self.update_config = self._load_update_config()
    
    def _load_update_config(self) -> Dict[str, Any]:
        """Load update configuration from file or use defaults."""
        config = {
            "auto_update": False,
            "update_frequency": "manual",
            "last_update": None,
            "update_days": 7
        }
        
        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    file_config = json.load(f)
                    config.update(file_config.get("semantic_search", {}).get("update_config", {}))
            except Exception as e:
                logger.warning(f"Error loading update config: {e}")
        
        return config
    
    def _save_update_config(self) -> None:
        """Save update configuration to file."""
        if not self.config_path:
            return
        
        config_dir = Path(self.config_path).parent
        config_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing config or create new one
        full_config = {}
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    full_config = json.load(f)
            except Exception:
                pass
        
        # Update semantic search config
        if "semantic_search" not in full_config:
            full_config["semantic_search"] = {}
        
        full_config["semantic_search"]["update_config"] = self.update_config
        
        try:
            with open(self.config_path, 'w') as f:
                json.dump(full_config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving update config: {e}")
    
    def _process_item_fulltext(self, item: Dict[str, Any]) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
        """
        Process fulltext content for a Zotero item.
        
        Args:
            item: Zotero item dictionary
            
        Returns:
            Tuple of (documents, metadatas, ids) for fulltext chunks
        """        
        item_key = item.get("key", "")
        if not item_key:
            return [], [], []
        
        # Get attachment details
        fulltext = self._get_fulltext(item)
        if not fulltext or not fulltext.strip():
            logger.debug(f"No suitable attachment text found for item {item_key}")
            return [], [], []
        
        try:
            # Prepare documents and metadata
            documents = []
            metadatas = []
            ids = []

            base_metadata = self._create_metadata(item)

            # Chunk the text
            chunks_data, offsets = self._create_chunks(fulltext)
            
            for i, (chunk, offset) in enumerate(zip(chunks_data, offsets)):

                # Create chunk-specific metadata
                chunk_id =  f"{item_key}_fulltext_{i}"
                chunk_metadata = base_metadata.copy()
                chunk_metadata.update({
                    "chunk_id": chunk_id,
                    "chunk_offset": str(offset),
                    "chunk_total": len(chunk),
                    "content_type": "fulltext"
                })
                
                documents.append(chunk)
                metadatas.append(chunk_metadata)
                ids.append(chunk_id)
            
            logger.info(f"Created {len(documents)} fulltext chunks for item {item_key}")
            return documents, metadatas, ids
            
        except Exception as e:
            logger.error(f"Error processing fulltext for item {item_key}: {e}")
            return [], [], []
    
    def _get_fulltext(self, item: dict) -> str|None:
        """
        Get fulltext content for a Zotero item.
        
        Args:
            item_key: Zotero item key
            
        Returns:
            Fulltext content as string
        """
        item_key = item.get("key", "")
                
        # Try to get attachment details
        attachment = get_attachment_details(self.zotero_client, item)
        if not attachment:
            logger.error(f"No attachment found for item with key: {item_key}")
            return None
                
        # Try fetching full text from Zotero's full text index first
        try:
            full_text_data = self.zotero_client.fulltext_item(attachment.key)
            if full_text_data and "content" in full_text_data and full_text_data["content"]:
                logger.debug("Successfully retrieved full text from Zotero's index")
                return full_text_data['content']
        except Exception as fulltext_error:
            logger.error(f"Couldn't retrieve indexed full text for item with key: {item_key}: {str(fulltext_error)}")
        
        # If we couldn't get indexed full text, try to download and convert the file
        try:
            logger.debug(f"Attempting to download and convert attachment {attachment.key} for item with key: {item_key}")
            
            # Download the file to a temporary location
            import tempfile
            import os
            
            with tempfile.TemporaryDirectory() as tmpdir:
                file_path = os.path.join(tmpdir, attachment.filename or f"{attachment.key}.pdf")
                self.zotero_client.dump(attachment.key, filename=os.path.basename(file_path), path=tmpdir)
                
                if os.path.exists(file_path):
                    logger.debug(f"Downloaded file to {file_path} for item with key: {item_key}, converting to markdown")
                    converted_text = convert_to_markdown(file_path)
                    return converted_text
                else:
                    logger.error(f"File download failed for item with key: {item_key}.")
                    return None
        except Exception as e:
            logger.error(f"Error downloading or converting attachment for item with key {item_key}: {e}")
            return None
                

    def _create_chunks(self, text) -> Tuple[List[str], List[Tuple[int, int]]]:
        # Roughly 750 words per chunk, with 250 word overlap
        chunk_size = 1000
        overlap = 250
        chunker = semchunk.chunkerify('cl100k_base', chunk_size)

        # Pass an `offsets` argument to return the offsets of chunks, as well as an `overlap`
        # argument to overlap chunks by a ratio (if < 1) or an absolute number of tokens (if >= 1).
        chunks, offsets = chunker(text, offsets = True, overlap = overlap)

        return chunks, offsets

    def _create_metadata(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create metadata for a Zotero item.
        
        Args:
            item: Zotero item dictionary
            
        Returns:
            Metadata dictionary for ChromaDB
        """
        data = item.get("data", {})
        
        metadata = {
            "item_key": item.get("key", ""),
            "item_type": data.get("itemType", ""),
            "title": data.get("title", ""),
            "date": data.get("date", ""),
            "date_added": data.get("dateAdded", ""),
            "date_modified": data.get("dateModified", ""),
            "creators": format_creators(data.get("creators", [])),
            "publication": data.get("publicationTitle", ""),
            "url": data.get("url", ""),
            "doi": data.get("DOI", ""),
        }
        
        # Add tags as a single string
        if tags := data.get("tags"):
            metadata["tags"] = " ".join([tag.get("tag", "") for tag in tags])
        else:
            metadata["tags"] = ""
        
        # Add citation key if available
        extra = data.get("extra", "")
        citation_key = ""
        for line in extra.split("\n"):
            if line.lower().startswith(("citation key:", "citationkey:")):
                citation_key = line.split(":", 1)[1].strip()
                break
        metadata["citation_key"] = citation_key
        
        return metadata
    
    def should_update_database(self) -> bool:
        """Check if the database should be updated based on configuration."""
        if not self.update_config.get("auto_update", False):
            return False
        
        frequency = self.update_config.get("update_frequency", "manual")
        
        if frequency == "manual":
            return False
        elif frequency == "startup":
            return True
        elif frequency == "daily":
            last_update = self.update_config.get("last_update")
            if not last_update:
                return True
            
            last_update_date = datetime.fromisoformat(last_update)
            return datetime.now() - last_update_date >= timedelta(days=1)
        elif frequency.startswith("every_"):
            try:
                days = int(frequency.split("_")[1])
                last_update = self.update_config.get("last_update")
                if not last_update:
                    return True
                
                last_update_date = datetime.fromisoformat(last_update)
                return datetime.now() - last_update_date >= timedelta(days=days)
            except (ValueError, IndexError):
                return False
        
        return False
    
    def update_database(self, 
                       force_full_rebuild: bool = False,
                       limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Update the semantic search database with Zotero items.
        
        Args:
            force_full_rebuild: Whether to rebuild the entire database
            limit: Limit number of items to process (for testing)
            
        Returns:
            Update statistics
        """
        logger.info("Starting database update...")
        start_time = datetime.now()
        
        stats = {
            "total_items": 0,
            "processed_items": 0,
            "added_items": 0,
            "updated_items": 0,
            "skipped_items": 0,
            "errors": 0,
            "start_time": start_time.isoformat(),
            "duration": None
        }
        
        try:
            # Reset collection if force rebuild
            if force_full_rebuild:
                logger.info("Force rebuilding database...")
                self.chroma_client.reset_collection()
            
            # Get all items from Zotero
            logger.info("Fetching items from Zotero...")
            
            # Fetch items in batches to handle large libraries
            batch_size = 100
            start = 0
            all_items = []
            
            while True:
                batch_params = {"start": start, "limit": batch_size, "itemType": "-attachment"}
                if limit and len(all_items) >= limit:
                    break
                
                items = self.zotero_client.items(**batch_params)
                if not items:
                    break
                
                # Filter out attachments and notes by default
                filtered_items = [
                    item for item in items 
                    if item.get("data", {}).get("itemType") not in ["attachment", "note"]
                ]
                
                all_items.extend(filtered_items)
                start += batch_size
                
                if len(all_items) >= batch_size:
                    break
            
            if limit:
                all_items = all_items[:limit]
            
            stats["total_items"] = len(all_items)
            logger.info(f"Found {stats['total_items']} items to process")
            
            # Process items in batches
            batch_size = 50
            for i in range(0, len(all_items), batch_size):
                batch = all_items[i:i + batch_size]
                batch_stats = self._process_item_batch(batch, force_full_rebuild)
                
                stats["processed_items"] += batch_stats["processed"]
                stats["added_items"] += batch_stats["added"]
                stats["updated_items"] += batch_stats["updated"]
                stats["skipped_items"] += batch_stats["skipped"]
                stats["errors"] += batch_stats["errors"]
                
                logger.info(f"Processed {stats['processed_items']}/{stats['total_items']} items")
            
            # Update last update time
            self.update_config["last_update"] = datetime.now().isoformat()
            self._save_update_config()
            
            end_time = datetime.now()
            stats["duration"] = str(end_time - start_time)
            stats["end_time"] = end_time.isoformat()
            
            logger.info(f"Database update completed in {stats['duration']}")
            return stats
            
        except Exception as e:
            logger.error(f"Error updating database: {e}")
            stats["error"] = str(e)
            end_time = datetime.now()
            stats["duration"] = str(end_time - start_time)
            return stats
    
    def _process_item_batch(self, items: List[Dict[str, Any]], force_rebuild: bool = False) -> Dict[str, int]:
        """Process a batch of items."""
        stats = {"processed": 0, "added": 0, "updated": 0, "skipped": 0, "errors": 0}
        
        documents = []
        metadatas = []
        ids = []
        
        for item in items:
            try:
                item_key = item.get("key", "")
                if not item_key:
                    stats["skipped"] += 1
                    continue
                
                # Check if item exists and needs update
                if not force_rebuild and self.chroma_client.document_exists(item_key):
                    # For now, skip existing items (could implement update logic here)
                    stats["skipped"] += 1
                    continue
                
                chunk_texts, chunk_metadatas, chunk_ids = self._process_item_fulltext(item)
                
                if len(chunk_texts) == 0:
                    stats["skipped"] += 1
                    continue
                
                documents.extend(chunk_texts)
                metadatas.extend(chunk_metadatas)
                ids.extend(chunk_ids)
                
                stats["processed"] += 1
                
            except Exception as e:
                logger.error(f"Error processing item {item.get('key', 'unknown')}: {e}")
                stats["errors"] += 1
        
        # Add documents to ChromaDB if any
        if documents:
            try:
                self.chroma_client.upsert_documents(documents, metadatas, ids)
                stats["added"] += len(documents)
            except Exception as e:
                logger.error(f"Error adding documents to ChromaDB: {e}")
                stats["errors"] += len(documents)
        
        return stats
    
    def search(self, 
               query: str, 
               limit: int = 10,
               filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform semantic search over the Zotero library.
        
        Args:
            query: Search query text
            limit: Maximum number of results to return
            filters: Optional metadata filters
            
        Returns:
            Search results with Zotero item details
        """
        try:
            # Perform semantic search
            results = self.chroma_client.search(
                query_texts=[query],
                n_results=limit,
                where=filters
            )
            
            # Enrich results with full Zotero item data
            enriched_results = self._enrich_search_results(results, query)
            
            return {
                "query": query,
                "limit": limit,
                "filters": filters,
                "results": enriched_results,
                "total_found": len(enriched_results)
            }
            
        except Exception as e:
            logger.error(f"Error performing semantic search: {e}")
            return {
                "query": query,
                "limit": limit,
                "filters": filters,
                "results": [],
                "total_found": 0,
                "error": str(e)
            }
    
    def _enrich_search_results(self, chroma_results: Dict[str, Any], query: str) -> List[Dict[str, Any]]:
        """Enrich ChromaDB results with full Zotero item data."""
        enriched = []
        
        if not chroma_results.get("ids") or not chroma_results["ids"][0]:
            return enriched
        
        ids = chroma_results["ids"][0]
        distances = chroma_results.get("distances", [[]])[0]
        documents = chroma_results.get("documents", [[]])[0]
        metadatas = chroma_results.get("metadatas", [[]])[0]
        
        for i, item_key in enumerate(ids):
            try:
                # Get full item data from Zotero
                zotero_item = self.zotero_client.item(item_key)
                
                enriched_result = {
                    "item_key": item_key,
                    "similarity_score": 1 - distances[i] if i < len(distances) else 0,
                    "matched_text": documents[i] if i < len(documents) else "",
                    "metadata": metadatas[i] if i < len(metadatas) else {},
                    "zotero_item": zotero_item,
                    "query": query
                }
                
                enriched.append(enriched_result)
                
            except Exception as e:
                logger.error(f"Error enriching result for item {item_key}: {e}")
                # Include basic result even if enrichment fails
                enriched.append({
                    "item_key": item_key,
                    "similarity_score": 1 - distances[i] if i < len(distances) else 0,
                    "matched_text": documents[i] if i < len(documents) else "",
                    "metadata": metadatas[i] if i < len(metadatas) else {},
                    "query": query,
                    "error": f"Could not fetch full item data: {e}"
                })
        
        return enriched
    
    def get_database_status(self) -> Dict[str, Any]:
        """Get status information about the semantic search database."""
        collection_info = self.chroma_client.get_collection_info()
        
        return {
            "collection_info": collection_info,
            "update_config": self.update_config,
            "should_update": self.should_update_database(),
            "last_update": self.update_config.get("last_update"),
        }
    
    def delete_item(self, item_key: str) -> bool:
        """Delete an item from the semantic search database."""
        try:
            self.chroma_client.delete_documents([item_key])
            return True
        except Exception as e:
            logger.error(f"Error deleting item {item_key}: {e}")
            return False


def create_semantic_search(config_path: Optional[str] = None) -> ZoteroSemanticSearch:
    """
    Create a ZoteroSemanticSearch instance.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configured ZoteroSemanticSearch instance
    """
    return ZoteroSemanticSearch(config_path=config_path)