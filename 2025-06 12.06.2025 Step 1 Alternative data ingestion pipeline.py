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
from llama_index.core import Settings

# Configure global embedding model
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Download required NLTK data
'''Ensure that the NLTK punkt tokenizer is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
'''

FILE_PATH = r"C:\\Users\\kuzne\\Desktop\\03 University\\Dissertation\\06 Medical Text Corpus\\BMJ\\Stripped PDFs\\Alzheimer's disease stripped.pdf"

# Path to the alternative Chroma database 
DATABASE_PATH = r"C:\\Users\\kuzne\\Documents\\Python_repo\\2025_01_dissertation\\2025_dissertation\\chromadb_alternative"

PROCESSED_FILES_LOG = r"C:\\Users\\kuzne\\Desktop\\03 University\\Dissertation\\06 Medical Text Corpus\\BMJ\\Stripped PDFs\\processed_files.json"


'''
Step 1. Load PDFs and parse them into semantic chunks.'''


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
    Tokenize text into sentences using NLTK. 
    
    Args:
        text: Input text to tokenize
        
    Returns:
        List of sentences

    1. Uses NLTK's sentence tokenizer to split the text into sentences.
    2. Get rid of empty sentences and in-text references like [1].
    https://www.nltk.org/api/nltk.tokenize.sent_tokenize.html
    """
    sentences = nltk.sent_tokenize(text)
    # Clean sentences: remove empty ones and strip whitespace
    sentences = [sent.strip() for sent in sentences if sent.strip()]
    # Filter out sentences with citations
    citation_pattern = re.compile(r'\[\d+\]')
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

def semantic_parse_nodes(file_path, 
                        similarity_threshold: float = 0.5,  # Lowered from 0.8
                        max_chunk_chars: int = 800,  
                        min_chunk_sentences: int = 2,  
                        min_chunk_chars: int = 150,  
                        embedding_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2') -> List[TextNode]:
    """
    Parse documents into semantic chunks using sentence-level similarity
    
    Args:
        document: Document object from LlamaIndex. Use load_pdf() function. 
        similarity_threshold: Cosine similarity threshold (lowered to 0.5)
        max_chunk_chars: Maximum characters per chunk (increased to 800) - 
        min_chunk_sentences: Minimum sentences per chunk
        min_chunk_chars: Minimum characters per chunk
        embedding_model_name: Name of the HuggingFace embedding model
        
    Returns:
        List of TextNode objects
    """
    # Initialize embedding model
    embed_model = HuggingFaceEmbedding(model_name=embedding_model_name)

    document = load_pdf(file_path)
    all_nodes = []
    # Step 1: Tokenize into sentences
    sentences = tokenize_sentences(document.text)
    print(f"  Tokenized into {len(sentences)} sentences")
        
        
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
                    'chunk_index': chunk_idx,
                    'semantic_metadata': metadata
                }
            )
        all_nodes.append(node)
    
    print(f"Total nodes created: {len(all_nodes)}")
    return all_nodes


def initialize_vector_store(database_path,
                            load_index=True):
    """
    Initialize the vector store and return the collection and index
    """
    try:    
        client = chromadb.PersistentClient(path=database_path)
        collection = client.get_or_create_collection(name="stripped_articles")

        # Get collection info
        collection_info = collection.count()
        print(f"Found collection with {collection_info} documents")

        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Only load the index if requested and collection is not empty
        index = None
        if load_index and collection_info > 0:
            print(f"Loading index into memory...")
            index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        
        return vector_store, storage_context, index
    
    except Exception as e:
        print(f"Error initializing vector store: {str(e)}")
        raise


def get_file_hash(file_path):
    """
    Generate a hash of file content to uniquely identify it
    """
    with open(file_path, 'rb') as f:
        file_hash = hashlib.md5(f.read()).hexdigest()

    return file_hash


def load_processed_files():
    """
    Load the list of already processed files
    """
    if os.path.exists(PROCESSED_FILES_LOG):
        with open(PROCESSED_FILES_LOG, 'r') as f:
            return json.load(f)
    return {}


def save_processed_files(processed_files):
    """
    Save the updated list of processed files
    """
    with open(PROCESSED_FILES_LOG, 'w') as f:
        json.dump(processed_files, f, indent=2)


def add_document_to_index(file_path, 
                          storage_context,  
                          similarity_threshold=0.5,  
                          max_chunk_chars=800,  
                          min_chunk_sentences=2,  
                          min_chunk_chars=150,  
                          similarity_embedding_model='sentence-transformers/all-MiniLM-L6-v2'):
    """
    Process a single PDF and add it to the index
    """
    file_name = os.path.basename(file_path)
    file_hash = get_file_hash(file_path)

    processed_files = load_processed_files()
    

    # Check if file was already processed
    if file_hash in processed_files:
        print(f"Skipping {file_name} - already processed previously")
        return False, True  # Not added, but not a failure

    try:
        nodes = semantic_parse_nodes(file_path, 
                                     similarity_threshold=similarity_threshold,  
                          max_chunk_chars=max_chunk_chars,  
                          min_chunk_sentences=min_chunk_sentences,  
                          min_chunk_chars=min_chunk_chars,  
                          embedding_model_name=similarity_embedding_model)
        
        # Variable created by initialize_vector_store function
        global index 
        for i, node in nodes:
            node.metadata['file_name'] = file_name
            node.node_id = f"{file_hash}_chunk_{i}"
            index.insert(node)
        print(f"Successfully inserted {len(nodes)} nodes")
        
        # Record this file as processed
        processed_files[file_hash] = {
            "file_name": file_name,
            "file_path": file_path,
            "date_added": str(Path(file_path).stat().st_mtime),
            "node_count": len(nodes)
        }
        save_processed_files(processed_files)
        print(f"Added document to vector store: {file_name}")
        return True, False  # Successfully added, not a failure
 
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False, False  # Not added, and it's a failure  
    

def process_folder(folder_path, 
                   storage_context, 
                   similarity_threshold=0.5,  # Lowered from 0.8
                          max_chunk_chars=800,  # Increased from 500
                          min_chunk_sentences=2,  # NEW: Force at least 2 sentences
                          min_chunk_chars=150,  # NEW: Force minimum character count
                          similarity_embedding_model='sentence-transformers/all-MiniLM-L6-v2'
                   
                   ):
    """
    Process all PDF files in a folder
    """
    added_count = 0
    skipped_count = 0
    failed_count = 0
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.pdf'):
            file_path = os.path.join(folder_path, filename)
            print(f"\nProcessing: {filename}")
            added, skipped = add_document_to_index(file_path=file_path, 
                                                   storage_context=storage_context,
                                                   similarity_threshold=similarity_threshold, 
                          max_chunk_chars=max_chunk_chars,  
                          min_chunk_sentences=min_chunk_sentences,  
                          min_chunk_chars=min_chunk_chars,  
                          similarity_embedding_model=similarity_embedding_model)
            
            if added:
                added_count += 1
            elif skipped:
                skipped_count += 1
            else:
                failed_count += 1
            
    
    print(f"\nProcessing complete.")
    print(f"Added: {added_count} documents")
    print(f"Skipped (already processed): {skipped_count} documents")
    print(f"Failed: {failed_count} documents")
    print(f"Vector store location: {DATABASE_PATH}")



vector_store, storage_context, index = initialize_vector_store(DATABASE_PATH, load_index=True)

process_folder(
    folder_path=r"C:\\Users\\kuzne\\Desktop\\03 University\\Dissertation\\06 Medical Text Corpus\\BMJ\\Stripped PDFs",
    storage_context=storage_context,
    similarity_threshold=0.5,  # Lowered from 0.8
    max_chunk_chars=800,  # Increased from 500
    min_chunk_sentences=2,  # NEW: Force at least 2 sentences
    min_chunk_chars=150,  # NEW: Force minimum character count
    similarity_embedding_model='sentence-transformers/all-MiniLM-L6-v2'
)