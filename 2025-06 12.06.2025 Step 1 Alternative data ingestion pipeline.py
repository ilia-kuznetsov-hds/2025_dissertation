import os
import nltk
import numpy as np
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from llama_index.readers.file import PyMuPDFReader
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
import chromadb
import hashlib
from pathlib import Path
import json
import re 

# Download required NLTK data
'''Ensure that the NLTK punkt tokenizer is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
'''

FILE_PATH = r"C:\\Users\\kuzne\\Desktop\\03 University\\Dissertation\\06 Medical Text Corpus\\BMJ\\Stripped PDFs\\Alzheimer's disease stripped.pdf"



def load_pdf(file_path):
    """
    Load PDF file and return Document objects
    """
    loader = PyMuPDFReader()
    documents = loader.load(file_path=file_path)
    print(f"Loaded {len(documents)} document(s) from {file_path}")
    return documents


def tokenize_sentences(text: str) -> List[str]:
    """
    Tokenize text into sentences using NLTK
    
    Args:
        text: Input text to tokenize
        
    Returns:
        List of sentences

    1. Uses NLTK's sentence tokenizer to split the text into sentences.
    https://www.nltk.org/api/nltk.tokenize.sent_tokenize.html
    """
    sentences = nltk.sent_tokenize(text)
    # Clean sentences: remove empty ones and strip whitespace
    sentences = [sent.strip() for sent in sentences if sent.strip()]

    citation_pattern = re.compile(r'\[\d+\]')
    # Filter out sentences with citations
    filtered_sentences = [citation_pattern.sub('', sent).strip() for sent in sentences]

    return filtered_sentences



def generate_sentence_embeddings(sentences: List[str], embed_model) -> np.ndarray:
    """
    Generate embeddings for a list of sentences
    
    Args:
        sentences: List of sentences to embed
        embed_model: HuggingFace embedding model
        
    Returns:
        numpy array of sentence embeddings
    """
    if not sentences:
        return np.array([])
    
    # Generate embeddings for all sentences
    embeddings = []
    for sentence in sentences:
        embedding = embed_model.get_text_embedding(sentence)
        embeddings.append(embedding)
    
    return np.array(embeddings)


def calculate_similarity_scores(embeddings: np.ndarray) -> List[float]:
    """
    Calculate cosine similarity between consecutive sentence embeddings
    
    Args:
        embeddings: Array of sentence embeddings
        
    Returns:
        List of similarity scores between consecutive sentences
    """
    if len(embeddings) < 2:
        return []
    
    similarities = []
    # iterate through embeddings except the last one since we compare pairs
    for i in range(len(embeddings) - 1):

        # Reshape for cosine_similarity function
        # This is necessary because cosine_similarity expects 2D arrays
        emb1 = embeddings[i].reshape(1, -1)
        emb2 = embeddings[i + 1].reshape(1, -1)
        similarity = cosine_similarity(emb1, emb2)[0][0]
        similarities.append(similarity)
    
    return similarities


def create_semantic_chunks(sentences: List[str], 
                          similarities: List[float], 
                          similarity_threshold: float = 0.5,  # Lowered from 0.8
                          max_chunk_chars: int = 500,
                          min_chunk_sentences: int = 2,  # NEW: Minimum sentences per chunk
                          min_chunk_chars: int = 100) -> List[Tuple[str, dict]]:  # NEW: Minimum characters
    """
    Group sentences into semantic chunks based on similarity scores

    Fucntion loops through each similarity score and evaluates whether to add the next
    sentence to the current chunk or start new one. 
    If addition of another sentence exceed the maximum character limit for a chunk,
    it will create a new chunk. 
    If the similarity score equals 0.8 (as in the paper ChunkRAG), each sentence will become 
    a separate chunk. When a threshold is lowered to 0.5, the function will create small chunks.

    
    Args:
        sentences: List of sentences
        similarities: List of similarity scores between consecutive sentences
        similarity_threshold: Threshold for creating new chunks (lowered to 0.5)
        max_chunk_chars: Maximum characters per chunk
        min_chunk_sentences: Minimum sentences per chunk
        min_chunk_chars: Minimum characters per chunk
        
    Returns:
        List of tuples (chunk_text, metadata)
    """
    if not sentences:
        return []
    
    chunks = []
    current_chunk_sentences = [sentences[0]]
    current_chunk_chars = len(sentences[0])
    
    for i, similarity in enumerate(similarities):
        next_sentence = sentences[i + 1]
        next_sentence_chars = len(next_sentence)
        
        # Chunking logic
        will_exceed_max = current_chunk_chars + next_sentence_chars > max_chunk_chars
        below_similarity = similarity < similarity_threshold
        meets_min_requirements = (
            len(current_chunk_sentences) >= min_chunk_sentences and 
            current_chunk_chars >= min_chunk_chars
        )
        
        # Only create new chunk if we have sufficient content AND hit a boundary
        should_create_new_chunk = (
            will_exceed_max or  # Hard size limit
            (below_similarity and meets_min_requirements)  # Semantic boundary + minimum size
        )
        
        if should_create_new_chunk and meets_min_requirements:
            # Finalize current chunk
            chunk_text = ' '.join(current_chunk_sentences)
            chunk_metadata = {
                'sentence_count': len(current_chunk_sentences),
                'char_count': len(chunk_text)
            }
            chunks.append((chunk_text, chunk_metadata))
            
            # Start new chunk
            current_chunk_sentences = [next_sentence]
            current_chunk_chars = next_sentence_chars
        else:
            # Add to current chunk (either too small or within similarity threshold)
            current_chunk_sentences.append(next_sentence)
            current_chunk_chars += next_sentence_chars
    
    # Add the last chunk (always include it, even if small)
    if current_chunk_sentences:
        chunk_text = ' '.join(current_chunk_sentences)
        chunk_metadata = {
            'sentence_count': len(current_chunk_sentences),
            'char_count': len(chunk_text)
        }
        chunks.append((chunk_text, chunk_metadata))
    
    return chunks

def semantic_parse_nodes(documents, 
                        similarity_threshold: float = 0.5,  # Lowered from 0.8
                        max_chunk_chars: int = 800,  
                        min_chunk_sentences: int = 2,  
                        min_chunk_chars: int = 150,  
                        embedding_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2') -> List[TextNode]:
    """
    Parse documents into semantic chunks using sentence-level similarity
    
    Args:
        documents: List of Document objects from LlamaIndex
        similarity_threshold: Cosine similarity threshold (lowered to 0.5)
        max_chunk_chars: Maximum characters per chunk (increased to 800)
        min_chunk_sentences: Minimum sentences per chunk
        min_chunk_chars: Minimum characters per chunk
        embedding_model_name: Name of the HuggingFace embedding model
        
    Returns:
        List of TextNode objects
    """
    # Initialize embedding model
    embed_model = HuggingFaceEmbedding(model_name=embedding_model_name)
    
    all_nodes = []
    
    for doc_idx, document in enumerate(documents):
        print(f"Processing document {doc_idx + 1}/{len(documents)}")
        
        # Step 1: Tokenize into sentences
        sentences = tokenize_sentences(document.text)
        print(f"  Tokenized into {len(sentences)} sentences")
        
        if len(sentences) == 0:
            continue
        
        # Step 2: Generate sentence embeddings
        print("  Generating sentence embeddings...")
        sentence_embeddings = generate_sentence_embeddings(sentences, embed_model)
        
        # Step 3: Calculate similarity scores
        similarities = calculate_similarity_scores(sentence_embeddings)
        print(f"  Calculated {len(similarities)} similarity scores")
        
        # Print similarity statistics for debugging
        if similarities:
            print(f"  Similarity stats: min={min(similarities):.3f}, max={max(similarities):.3f}, mean={np.mean(similarities):.3f}")
        
        # Step 4: Create semantic chunks
        chunks = create_semantic_chunks(
            sentences, 
            similarities, 
            similarity_threshold, 
            max_chunk_chars,
            min_chunk_sentences,
            min_chunk_chars
        )
        print(f"  Created {len(chunks)} semantic chunks")
        
        # Print chunk size statistics
        chunk_sizes = [metadata['sentence_count'] for _, metadata in chunks]
        print(f"  Chunk sizes: min={min(chunk_sizes)}, max={max(chunk_sizes)}, mean={np.mean(chunk_sizes):.1f}")
        
        # Step 5: Convert to TextNode objects
        for chunk_idx, (chunk_text, metadata) in enumerate(chunks):
            node = TextNode(
                text=chunk_text,
                metadata={
                    **document.metadata,
                    'doc_index': doc_idx,
                    'chunk_index': chunk_idx,
                    'semantic_metadata': metadata
                }
            )
            all_nodes.append(node)
    
    print(f"Total nodes created: {len(all_nodes)}")
    return all_nodes

# Process the PDF with updated parameters
nodes = semantic_parse_nodes(
    documents=load_pdf(FILE_PATH),
    similarity_threshold=0.5,  # Lowered from 0.8
    max_chunk_chars=800,  # Increased from 500
    min_chunk_sentences=2,  # NEW: Force at least 2 sentences
    min_chunk_chars=150,  # NEW: Force minimum character count
    embedding_model_name='sentence-transformers/all-MiniLM-L6-v2'
)

print(nodes[:3])  # Print first 3 nodes for verification

