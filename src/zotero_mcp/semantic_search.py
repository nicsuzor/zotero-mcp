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

from pyzotero import zotero

from .chroma_client import ChromaClient, create_chroma_client
from .client import get_zotero_client, get_attachment_details, convert_to_markdown
from .utils import format_creators
from .text_processor import AcademicTextChunker, extract_pdf_annotations, process_annotations_for_search

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
        
        # Load fulltext configuration
        self.fulltext_config = self._load_fulltext_config()
        
        # Initialize text chunker if fulltext is enabled
        self.text_chunker = None
        if self.fulltext_config.get("enabled", False):
            self.text_chunker = AcademicTextChunker(
                chunk_size=self.fulltext_config.get("chunk_size", 1000),
                chunk_overlap=self.fulltext_config.get("chunk_overlap", 200),
                max_chunks_per_document=self.fulltext_config.get("max_chunks_per_item", 50)
            )
    
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
    
    def _load_fulltext_config(self) -> Dict[str, Any]:
        """Load fulltext configuration from file or use defaults."""
        config = {
            "enabled": False,
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "max_chunks_per_item": 50,
            "include_annotations": True
        }
        
        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    file_config = json.load(f)
                    config.update(file_config.get("semantic_search", {}).get("fulltext", {}))
            except Exception as e:
                logger.warning(f"Error loading fulltext config: {e}")
        
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
    
    def _create_document_text(self, item: Dict[str, Any]) -> str:
        """
        Create searchable text from a Zotero item.
        
        Args:
            item: Zotero item dictionary
            
        Returns:
            Combined text for embedding
        """
        data = item.get("data", {})
        
        # Extract key fields for semantic search
        title = data.get("title", "")
        abstract = data.get("abstractNote", "")
        
        # Format creators as text
        creators = data.get("creators", [])
        creators_text = format_creators(creators)
        
        # Additional searchable content
        extra_fields = []
        
        # Publication details
        if publication := data.get("publicationTitle"):
            extra_fields.append(publication)
        
        # Tags
        if tags := data.get("tags"):
            tag_text = " ".join([tag.get("tag", "") for tag in tags])
            extra_fields.append(tag_text)
        
        # Note content (if available)
        if note := data.get("note"):
            # Clean HTML from notes
            import re
            note_text = re.sub(r'<[^>]+>', '', note)
            extra_fields.append(note_text)
        
        # Combine all text fields
        text_parts = [title, creators_text, abstract] + extra_fields
        return " ".join(filter(None, text_parts))
    
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
    
    def _process_item_fulltext(self, item: Dict[str, Any]) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
        """
        Process fulltext content for a Zotero item.
        
        Args:
            item: Zotero item dictionary
            
        Returns:
            Tuple of (documents, metadatas, ids) for fulltext chunks
        """
        if not self.text_chunker:
            return [], [], []
        
        item_key = item.get("key", "")
        if not item_key:
            return [], [], []
        
        # Get attachment details
        attachment = get_attachment_details(self.zotero_client, item)
        if not attachment:
            logger.debug(f"No suitable attachment found for item {item_key}")
            return [], [], []
        
        # Check if we've already processed this attachment
        if self._is_fulltext_processed(item_key, attachment.key):
            logger.debug(f"Fulltext already processed for {item_key}")
            return [], [], []
        
        try:
            # Extract fulltext using existing infrastructure
            full_text = self._extract_fulltext(attachment)
            if not full_text or not full_text.strip():
                logger.debug(f"No fulltext content extracted for {item_key}")
                return [], [], []
            
            # Chunk the text
            chunks = self.text_chunker.chunk_text(full_text, item.get("data", {}).get("title", "Document"))
            
            if not chunks:
                return [], [], []
            
            # Prepare documents and metadata
            documents = []
            metadatas = []
            ids = []
            
            base_metadata = self._create_metadata(item)
            
            for chunk in chunks:
                # Create chunk-specific metadata
                chunk_metadata = base_metadata.copy()
                chunk_metadata.update({
                    "chunk_index": chunk.chunk_index,
                    "chunk_total": chunk.chunk_total,
                    "section_title": chunk.section_title or "Content",
                    "attachment_key": attachment.key,
                    "content_type": "fulltext",
                    "token_count": chunk.token_count or 0
                })
                
                # Create unique ID for this chunk
                chunk_id = f"{item_key}_fulltext_{chunk.chunk_index}"
                
                documents.append(chunk.text)
                metadatas.append(chunk_metadata)
                ids.append(chunk_id)
            
            logger.info(f"Created {len(chunks)} fulltext chunks for item {item_key}")
            return documents, metadatas, ids
            
        except Exception as e:
            logger.error(f"Error processing fulltext for item {item_key}: {e}")
            return [], [], []
    
    def _process_item_annotations(self, item: Dict[str, Any]) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
        """
        Process PDF annotations for a Zotero item.
        
        Args:
            item: Zotero item dictionary
            
        Returns:
            Tuple of (documents, metadatas, ids) for annotations
        """
        if not self.fulltext_config.get("include_annotations", True):
            return [], [], []
        
        item_key = item.get("key", "")
        if not item_key:
            return [], [], []
        
        # Get attachment details
        attachment = get_attachment_details(self.zotero_client, item)
        if not attachment or attachment.content_type != "application/pdf":
            return [], [], []
        
        # Check if we've already processed annotations for this attachment
        if self._are_annotations_processed(item_key, attachment.key):
            logger.debug(f"Annotations already processed for {item_key}")
            return [], [], []
        
        try:
            # Download and extract annotations
            annotations_text = self._extract_annotations(attachment)
            if not annotations_text or not annotations_text.strip():
                logger.debug(f"No annotations found for {item_key}")
                return [], [], []
            
            # Create metadata
            base_metadata = self._create_metadata(item)
            annotation_metadata = base_metadata.copy()
            annotation_metadata.update({
                "attachment_key": attachment.key,
                "content_type": "annotations"
            })
            
            # Create unique ID for annotations
            annotation_id = f"{item_key}_annotations"
            
            logger.info(f"Processed annotations for item {item_key}")
            return [annotations_text], [annotation_metadata], [annotation_id]
            
        except Exception as e:
            logger.error(f"Error processing annotations for item {item_key}: {e}")
            return [], [], []
    
    def _extract_fulltext(self, attachment) -> str:
        """Extract fulltext from an attachment using existing infrastructure."""
        try:
            # Try Zotero's fulltext index first
            try:
                full_text_data = self.zotero_client.fulltext_item(attachment.key)
                if full_text_data and "content" in full_text_data and full_text_data["content"]:
                    return full_text_data["content"]
            except Exception:
                pass
            
            # Fallback to downloading and converting
            import tempfile
            import os
            
            with tempfile.TemporaryDirectory() as tmpdir:
                file_path = os.path.join(tmpdir, attachment.filename or f"{attachment.key}.pdf")
                self.zotero_client.dump(attachment.key, filename=os.path.basename(file_path), path=tmpdir)
                
                if os.path.exists(file_path):
                    return convert_to_markdown(file_path)
            
            return ""
            
        except Exception as e:
            logger.error(f"Error extracting fulltext from attachment {attachment.key}: {e}")
            return ""
    
    def _extract_annotations(self, attachment) -> str:
        """Extract annotations from a PDF attachment."""
        try:
            import tempfile
            import os
            
            with tempfile.TemporaryDirectory() as tmpdir:
                file_path = os.path.join(tmpdir, attachment.filename or f"{attachment.key}.pdf")
                self.zotero_client.dump(attachment.key, filename=os.path.basename(file_path), path=tmpdir)
                
                if os.path.exists(file_path):
                    annotations = extract_pdf_annotations(file_path)
                    return process_annotations_for_search(annotations)
            
            return ""
            
        except Exception as e:
            logger.error(f"Error extracting annotations from attachment {attachment.key}: {e}")
            return ""
    
    def _is_fulltext_processed(self, item_key: str, attachment_key: str) -> bool:
        """Check if fulltext has been processed for this item."""
        try:
            # Check if any fulltext chunks exist for this item
            return self.chroma_client.document_exists(f"{item_key}_fulltext_0", "fulltext")
        except Exception:
            return False
    
    def _are_annotations_processed(self, item_key: str, attachment_key: str) -> bool:
        """Check if annotations have been processed for this item."""
        try:
            return self.chroma_client.document_exists(f"{item_key}_annotations", "annotations")
        except Exception:
            return False
    
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
                batch_params = {"start": start, "limit": batch_size}
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
                
                if len(items) < batch_size:
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
        """Process a batch of items including fulltext and annotations."""
        stats = {
            "processed": 0, "added": 0, "updated": 0, "skipped": 0, "errors": 0,
            "fulltext_chunks": 0, "annotations_processed": 0
        }
        
        # Separate collections for different content types
        collections_data = {
            "items": {"documents": [], "metadatas": [], "ids": []},
            "fulltext": {"documents": [], "metadatas": [], "ids": []},
            "annotations": {"documents": [], "metadatas": [], "ids": []}
        }
        
        for item in items:
            try:
                item_key = item.get("key", "")
                if not item_key:
                    stats["skipped"] += 1
                    continue
                
                # Process metadata (items collection)
                if not force_rebuild and self.chroma_client.document_exists(item_key, "items"):
                    stats["skipped"] += 1
                else:
                    # Create document text and metadata
                    doc_text = self._create_document_text(item)
                    metadata = self._create_metadata(item)
                    
                    if doc_text.strip():
                        collections_data["items"]["documents"].append(doc_text)
                        collections_data["items"]["metadatas"].append(metadata)
                        collections_data["items"]["ids"].append(item_key)
                
                # Process fulltext if enabled
                if self.fulltext_config.get("enabled", False):
                    fulltext_docs, fulltext_metas, fulltext_ids = self._process_item_fulltext(item)
                    if fulltext_docs:
                        collections_data["fulltext"]["documents"].extend(fulltext_docs)
                        collections_data["fulltext"]["metadatas"].extend(fulltext_metas)
                        collections_data["fulltext"]["ids"].extend(fulltext_ids)
                        stats["fulltext_chunks"] += len(fulltext_docs)
                
                # Process annotations if enabled
                if self.fulltext_config.get("enabled", False) and self.fulltext_config.get("include_annotations", True):
                    annotation_docs, annotation_metas, annotation_ids = self._process_item_annotations(item)
                    if annotation_docs:
                        collections_data["annotations"]["documents"].extend(annotation_docs)
                        collections_data["annotations"]["metadatas"].extend(annotation_metas)
                        collections_data["annotations"]["ids"].extend(annotation_ids)
                        stats["annotations_processed"] += len(annotation_docs)
                
                stats["processed"] += 1
                
            except Exception as e:
                logger.error(f"Error processing item {item.get('key', 'unknown')}: {e}")
                stats["errors"] += 1
        
        # Add documents to appropriate collections
        for collection_type, data in collections_data.items():
            if data["documents"]:
                try:
                    self.chroma_client.upsert_documents(
                        data["documents"], 
                        data["metadatas"], 
                        data["ids"],
                        collection_type=collection_type
                    )
                    stats["added"] += len(data["documents"])
                    logger.info(f"Added {len(data['documents'])} documents to {collection_type} collection")
                except Exception as e:
                    logger.error(f"Error adding documents to {collection_type} collection: {e}")
                    stats["errors"] += len(data["documents"])
        
        return stats
    
    def search(self, 
               query: str, 
               limit: int = 10,
               filters: Optional[Dict[str, Any]] = None,
               search_collections: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform unified semantic search over the Zotero library.
        
        Args:
            query: Search query text
            limit: Maximum number of results to return
            filters: Optional metadata filters
            search_collections: Collections to search ('items', 'fulltext', 'annotations'). 
                               If None, searches based on fulltext config.
            
        Returns:
            Unified search results with Zotero item details
        """
        try:
            if search_collections is None:
                # Default collections based on configuration
                search_collections = ["items"]
                if self.fulltext_config.get("enabled", False):
                    search_collections.extend(["fulltext", "annotations"])
            
            # Search across specified collections
            all_results = {}
            for collection_type in search_collections:
                try:
                    collection_results = self.chroma_client.search(
                        query_texts=[query],
                        n_results=limit,
                        where=filters,
                        collection_type=collection_type
                    )
                    all_results[collection_type] = collection_results
                except Exception as e:
                    logger.warning(f"Error searching {collection_type} collection: {e}")
                    all_results[collection_type] = {"ids": [[]], "distances": [[]], "documents": [[]], "metadatas": [[]]}
            
            # Merge and rank results
            unified_results = self._merge_collection_results(all_results, limit)
            
            # Enrich results with full Zotero item data
            enriched_results = self._enrich_unified_results(unified_results, query)
            
            return {
                "query": query,
                "limit": limit,
                "filters": filters,
                "search_collections": search_collections,
                "results": enriched_results,
                "total_found": len(enriched_results),
                "collection_counts": {
                    ctype: len(results.get("ids", [[]])[0])
                    for ctype, results in all_results.items()
                }
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
    
    def _merge_collection_results(self, all_results: Dict[str, Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        """
        Merge results from multiple collections and rank by relevance.
        
        Args:
            all_results: Results from each collection
            limit: Maximum number of results to return
            
        Returns:
            Merged and ranked results
        """
        combined_results = []
        
        for collection_type, results in all_results.items():
            if not results.get("ids") or not results["ids"][0]:
                continue
            
            ids = results["ids"][0]
            distances = results.get("distances", [[]])[0]
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            
            for i, result_id in enumerate(ids):
                similarity_score = 1 - distances[i] if i < len(distances) else 0
                
                # For fulltext chunks, extract the base item key
                if collection_type == "fulltext" and "_fulltext_" in result_id:
                    item_key = result_id.split("_fulltext_")[0]
                elif collection_type == "annotations" and "_annotations" in result_id:
                    item_key = result_id.split("_annotations")[0]
                else:
                    item_key = result_id
                
                # Apply collection-specific scoring
                if collection_type == "fulltext":
                    # Boost fulltext matches slightly
                    similarity_score *= 1.1
                elif collection_type == "annotations":
                    # Boost annotation matches
                    similarity_score *= 1.05
                
                combined_results.append({
                    "result_id": result_id,
                    "item_key": item_key,
                    "collection_type": collection_type,
                    "similarity_score": similarity_score,
                    "matched_text": documents[i] if i < len(documents) else "",
                    "metadata": metadatas[i] if i < len(metadatas) else {}
                })
        
        # Sort by similarity score (descending) and take top results
        combined_results.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        # Deduplicate by item_key while preserving best matches
        seen_items = set()
        deduplicated = []
        
        for result in combined_results:
            item_key = result["item_key"]
            if item_key not in seen_items:
                seen_items.add(item_key)
                deduplicated.append(result)
            elif len(deduplicated) < limit:
                # Keep additional matches from different collections for the same item
                # if we haven't reached the limit
                deduplicated.append(result)
        
        return deduplicated[:limit]
    
    def _enrich_unified_results(self, unified_results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Enrich unified search results with full Zotero item data."""
        enriched = []
        
        for result in unified_results:
            try:
                item_key = result["item_key"]
                
                # Get full item data from Zotero
                zotero_item = self.zotero_client.item(item_key)
                
                enriched_result = {
                    "item_key": item_key,
                    "result_id": result["result_id"],
                    "collection_type": result["collection_type"],
                    "similarity_score": result["similarity_score"],
                    "matched_text": result["matched_text"],
                    "metadata": result["metadata"],
                    "zotero_item": zotero_item,
                    "query": query
                }
                
                # Add collection-specific context
                if result["collection_type"] == "fulltext":
                    enriched_result["match_context"] = "Full text content"
                    if "section_title" in result["metadata"]:
                        enriched_result["match_context"] += f" (Section: {result['metadata']['section_title']})"
                elif result["collection_type"] == "annotations":
                    enriched_result["match_context"] = "PDF annotations"
                else:
                    enriched_result["match_context"] = "Item metadata"
                
                enriched.append(enriched_result)
                
            except Exception as e:
                logger.error(f"Error enriching result for item {result.get('item_key', 'unknown')}: {e}")
                # Include basic result even if enrichment fails
                enriched.append({
                    "item_key": result.get("item_key", ""),
                    "result_id": result.get("result_id", ""),
                    "collection_type": result.get("collection_type", "unknown"),
                    "similarity_score": result.get("similarity_score", 0),
                    "matched_text": result.get("matched_text", ""),
                    "metadata": result.get("metadata", {}),
                    "query": query,
                    "match_context": "Error loading item data",
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
        """Delete an item and all its related data from the semantic search database."""
        try:
            self.chroma_client.delete_item_from_all_collections(item_key)
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