"""
Text processing utilities for academic document chunking and analysis.

This module provides intelligent text chunking optimized for academic papers,
with support for section-aware splitting and semantic boundary detection.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger.warning("tiktoken not available, falling back to character-based chunking")


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata."""
    
    text: str
    chunk_index: int
    chunk_total: Optional[int] = None
    section_title: Optional[str] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    token_count: Optional[int] = None


class AcademicTextChunker:
    """
    Text chunker optimized for academic papers.
    
    Features:
    - Token-aware chunking with configurable size and overlap
    - Section-aware splitting (respects headers and paragraphs)
    - Semantic boundary detection (avoids splitting mid-sentence)
    - Preserves document structure for better search results
    """
    
    def __init__(self, 
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 max_chunks_per_document: int = 50,
                 encoding_name: str = "cl100k_base"):
        """
        Initialize the text chunker.
        
        Args:
            chunk_size: Target size in tokens per chunk
            chunk_overlap: Number of tokens to overlap between chunks
            max_chunks_per_document: Maximum chunks to create per document
            encoding_name: tiktoken encoding to use for token counting
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_chunks_per_document = max_chunks_per_document
        
        # Initialize tokenizer
        if TIKTOKEN_AVAILABLE:
            try:
                self.encoding = tiktoken.get_encoding(encoding_name)
            except Exception as e:
                logger.warning(f"Error loading tiktoken encoding {encoding_name}: {e}")
                self.encoding = None
        else:
            self.encoding = None
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.encoding:
            return len(self.encoding.encode(text))
        else:
            # Fallback: rough estimation (1 token ≈ 4 characters for English)
            return len(text) // 4
    
    def _extract_sections(self, text: str) -> List[Tuple[str, str, int]]:
        """
        Extract sections from academic text based on headers.
        
        Returns:
            List of (section_title, section_text, start_position) tuples
        """
        sections = []
        
        # Common academic section patterns
        section_patterns = [
            r'^#{1,6}\s+(.+?)$',  # Markdown headers
            r'^([A-Z][A-Z\s]{2,})\s*$',  # ALL CAPS headers
            r'^\d+\.?\s+([A-Z][^.]*?)$',  # Numbered sections
            r'^([A-Z][a-z]+(?: [A-Z][a-z]+)*)\s*$',  # Title Case headers
        ]
        
        combined_pattern = '|'.join(f'({pattern})' for pattern in section_patterns)
        
        lines = text.split('\n')
        current_section = "Introduction"
        current_text = []
        current_start = 0
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Check if this line is a section header
            match = re.match(combined_pattern, line_stripped, re.MULTILINE)
            if match and len(line_stripped) < 100:  # Headers shouldn't be too long
                # Save previous section
                if current_text:
                    section_text = '\n'.join(current_text).strip()
                    if section_text:
                        sections.append((current_section, section_text, current_start))
                
                # Start new section
                header_text = next(group for group in match.groups() if group)
                if header_text:
                    current_section = header_text.strip()
                current_text = []
                current_start = len('\n'.join(lines[:i+1])) + 1
            else:
                current_text.append(line)
        
        # Add final section
        if current_text:
            section_text = '\n'.join(current_text).strip()
            if section_text:
                sections.append((current_section, section_text, current_start))
        
        # If no sections found, treat entire text as one section
        if not sections:
            sections = [("Document", text.strip(), 0)]
        
        return sections
    
    def _find_split_point(self, text: str, target_pos: int) -> int:
        """
        Find the best position to split text near target_pos.
        
        Prefers to split at:
        1. Paragraph breaks (double newlines)
        2. Sentence endings
        3. Clause endings (commas, semicolons)
        4. Word boundaries
        """
        if target_pos >= len(text):
            return len(text)
        
        # Search window around target position
        search_window = min(200, len(text) - target_pos)
        start_search = max(0, target_pos - search_window // 2)
        end_search = min(len(text), target_pos + search_window // 2)
        search_text = text[start_search:end_search]
        
        # Priority order for split points
        split_patterns = [
            (r'\n\s*\n', 'paragraph'),  # Paragraph breaks
            (r'[.!?]\s+[A-Z]', 'sentence'),  # Sentence endings
            (r'[;:]\s+', 'clause'),  # Clause endings
            (r'[,]\s+', 'comma'),  # Comma breaks
            (r'\s+', 'word'),  # Word boundaries
        ]
        
        best_pos = target_pos
        best_priority = 999
        
        for i, (pattern, priority_name) in enumerate(split_patterns):
            matches = list(re.finditer(pattern, search_text))
            if matches:
                # Find match closest to target
                target_in_search = target_pos - start_search
                closest_match = min(matches, 
                                  key=lambda m: abs(m.start() - target_in_search))
                
                split_pos = start_search + closest_match.start()
                if priority_name == 'sentence':
                    # For sentences, split after the punctuation
                    split_pos += 1
                
                if i < best_priority:
                    best_pos = split_pos
                    best_priority = i
                    break
        
        return max(1, min(best_pos, len(text) - 1))
    
    def chunk_text(self, text: str, source_title: str = "Document") -> List[TextChunk]:
        """
        Chunk text into overlapping segments optimized for academic content.
        
        Args:
            text: The text to chunk
            source_title: Title of the source document
            
        Returns:
            List of TextChunk objects
        """
        if not text or not text.strip():
            return []
        
        # Extract sections from the text
        sections = self._extract_sections(text)
        chunks = []
        
        for section_title, section_text, section_start in sections:
            section_chunks = self._chunk_section(section_text, section_title, section_start)
            chunks.extend(section_chunks)
            
            # Respect max chunks limit
            if len(chunks) >= self.max_chunks_per_document:
                chunks = chunks[:self.max_chunks_per_document]
                logger.warning(f"Truncated to {self.max_chunks_per_document} chunks for {source_title}")
                break
        
        # Set chunk_total for all chunks
        for chunk in chunks:
            chunk.chunk_total = len(chunks)
        
        logger.info(f"Created {len(chunks)} chunks from {len(sections)} sections for {source_title}")
        return chunks
    
    def _chunk_section(self, text: str, section_title: str, section_start: int) -> List[TextChunk]:
        """Chunk a single section of text."""
        if not text.strip():
            return []
        
        chunks = []
        current_pos = 0
        chunk_index = 0
        
        while current_pos < len(text):
            # Calculate chunk boundaries
            chunk_start = current_pos
            
            # Target end position based on token count
            if self.encoding:
                # Count tokens from current position
                remaining_text = text[current_pos:]
                tokens_so_far = 0
                char_pos = 0
                
                for char_pos in range(len(remaining_text)):
                    test_text = remaining_text[:char_pos + 1]
                    tokens_so_far = self.count_tokens(test_text)
                    if tokens_so_far >= self.chunk_size:
                        break
                
                target_end = current_pos + char_pos
            else:
                # Fallback: character-based estimation
                target_end = current_pos + (self.chunk_size * 4)
            
            # Find optimal split point
            chunk_end = self._find_split_point(text, min(target_end, len(text)))
            
            # Extract chunk text
            chunk_text = text[chunk_start:chunk_end].strip()
            
            if chunk_text:
                token_count = self.count_tokens(chunk_text)
                
                chunk = TextChunk(
                    text=chunk_text,
                    chunk_index=chunk_index,
                    section_title=section_title,
                    start_char=section_start + chunk_start,
                    end_char=section_start + chunk_end,
                    token_count=token_count
                )
                chunks.append(chunk)
                chunk_index += 1
            
            # Calculate next position with overlap
            if chunk_end >= len(text):
                break
            
            # Calculate overlap position
            if self.chunk_overlap > 0 and self.encoding:
                # Find position that gives us the desired overlap in tokens
                overlap_start = chunk_end
                for i in range(min(chunk_end, self.chunk_overlap * 4), 0, -1):
                    overlap_pos = chunk_end - i
                    if overlap_pos > chunk_start:
                        overlap_text = text[overlap_pos:chunk_end]
                        if self.count_tokens(overlap_text) <= self.chunk_overlap:
                            overlap_start = overlap_pos
                            break
                
                current_pos = overlap_start
            else:
                # No overlap or fallback
                current_pos = chunk_end
        
        return chunks


def extract_pdf_annotations(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract annotations from a PDF file using the existing pdfannots infrastructure.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of annotation dictionaries
    """
    try:
        from .pdfannots_helper import extract_annotations_from_pdf
        return extract_annotations_from_pdf(pdf_path)
    except ImportError as e:
        logger.error(f"Could not import pdfannots_helper: {e}")
        return []
    except Exception as e:
        logger.error(f"Error extracting annotations from {pdf_path}: {e}")
        return []


def process_annotations_for_search(annotations: List[Dict[str, Any]]) -> str:
    """
    Process PDF annotations into searchable text.
    
    Args:
        annotations: List of annotation dictionaries from pdfannots2json
        
    Returns:
        Combined annotation text for embedding
    """
    if not annotations:
        return ""
    
    annotation_texts = []
    
    for annotation in annotations:
        # Extract text content from different annotation types
        content_parts = []
        
        # Highlighted text
        if highlight_text := annotation.get('highlightText'):
            content_parts.append(f"Highlight: {highlight_text}")
        
        # Annotation comments/notes
        if comment := annotation.get('comment'):
            content_parts.append(f"Note: {comment}")
        
        # Markup text (strikethrough, underline, etc.)
        if markup_text := annotation.get('markupText'):
            content_parts.append(f"Markup: {markup_text}")
        
        # Combine parts for this annotation
        if content_parts:
            annotation_text = " | ".join(content_parts)
            
            # Add page context if available
            if page_num := annotation.get('page'):
                annotation_text = f"[Page {page_num}] {annotation_text}"
            
            annotation_texts.append(annotation_text)
    
    # Combine all annotations
    return " || ".join(annotation_texts) if annotation_texts else ""