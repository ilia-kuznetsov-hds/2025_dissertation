import hashlib
import json
import os
from pathlib import Path

import chromadb
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PyMuPDFReader
from llama_index.vector_stores.chroma import ChromaVectorStore


EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


def configure_embedding_model():
    # Use the same embedding model for all nodes added to the vector store.
    # all-mpnet-base-v2 truncates long input text, so chunk size should stay below
    # the model's practical input limit.
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)


# Repository root
REPO_ROOT = Path(__file__).resolve().parents[1]
# Folder path to the directory with PDF files of medical texts' corpus
FOLDER_PATH = str(REPO_ROOT / "data" / "knowledge_base")
# Path to the Chroma database that stores all the embeddings
DATABASE_PATH = str(REPO_ROOT / "chromadb")
# Path to store the record of processed files
PROCESSED_FILES_LOG = str(REPO_ROOT / "data" / "knowledge_base" / "processed_files.json")

'''
Part 1. PDF loading and text chunking

This section converts a PDF article into LlamaIndex text nodes.

Workflow:
1. load_pdf()
   Reads a PDF file with PyMuPDFReader and returns LlamaIndex Document objects.

2. parse_nodes()
   Splits the extracted documents into smaller text nodes using SentenceSplitter.
   The chunk size and overlap can be configured.

3. process_pdf()
   Runs the full PDF-to-nodes workflow and prints a small preview of the first
   few generated nodes for inspection.
'''


def load_pdf(file_path):
    '''
    Load a PDF file and extract its text into LlamaIndex Document objects.

    PyMuPDFReader reads the PDF from disk and returns a list of Document
    objects. These documents can then be split into smaller text nodes for
    embedding and indexing in the vector database.

    https://pymupdf.readthedocs.io/en/latest/rag.html

    '''
    loader = PyMuPDFReader()
    documents = loader.load(file_path=file_path)
    print(f"Loaded {len(documents)} document(s) from {file_path}")
    return documents


def parse_nodes(documents, chunk_size=1024, chunk_overlap=20):
    '''
    Split LlamaIndex Document objects into smaller text nodes.

    SentenceSplitter creates chunks that are suitable for embedding and vector
    indexing. It tries to preserve sentence boundaries while respecting the
    configured chunk size and overlap.

    Args:
        documents: List of LlamaIndex Document objects extracted from a PDF.
        chunk_size: Maximum size of each text chunk.
        chunk_overlap: Number of overlapping tokens between adjacent chunks.

    Returns:
        A list of LlamaIndex TextNode objects.

    https://developers.llamaindex.ai/python/framework/module_guides/loading/node_parsers/modules/
    
    '''
    text_splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap)
    nodes = text_splitter.get_nodes_from_documents(documents)
    print(f"Created {len(nodes)} from the document(s)")
    return nodes


def process_pdf(file_path, chunk_size, chunk_overlap):
    """
    Load a PDF file and convert it into LlamaIndex text nodes.

    Document objects represent the extracted PDF content. Text nodes are smaller
    chunks of those documents that are suitable for embedding, vector indexing,
    and later retrieval in a RAG pipeline.

    https://developers.llamaindex.ai/python/framework/module_guides/loading/documents_and_nodes/
    """

    documents = load_pdf(file_path)
    nodes = parse_nodes(documents, chunk_size, chunk_overlap)
    for i, node in enumerate(nodes[:3]):
        print(f'Node {i}')
        print(f' {len(node.text)}')
        print(f' First 100 chars: {node.text[:100]}')
    if len(nodes) > 3:
        print(f"... and {len(nodes) - 3} more nodes")

    return nodes


'''
Part 2. Add text nodes to the Chroma vector store

This section prepares the persistent Chroma database and adds processed PDF
nodes to it.

Workflow:
1. initialize_vector_store()
   Opens or creates the local Chroma collection and connects it to LlamaIndex
   through a StorageContext.

2. get_file_hash()
   Computes a content hash for each PDF so already-processed files can be
   skipped on later runs.

3. load_processed_files() / save_processed_files()
   Read and update the JSON log that tracks which PDFs have already been added.

4. add_document_to_index()
   Converts one PDF into text nodes and adds those nodes to the vector store.

5. process_folder()
   Iterates through all PDF files in the knowledge base folder and indexes any
   files that have not already been processed.
'''


def initialize_vector_store(database_path):
    """
    Open the persistent Chroma vector store and connect it to LlamaIndex.

    Returns a StorageContext that tells LlamaIndex to store newly embedded text
    nodes in the local Chroma collection named "articles".

    https://developers.llamaindex.ai/python/framework/understanding/rag/storing/

    """
    client = chromadb.PersistentClient(path=database_path)
    collection = client.get_or_create_collection(name="articles")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    return storage_context


def get_file_hash(file_path):
    """
    Generate an MD5 hash from the PDF file contents.

    The hash is used as a stable identifier for the file, so the ingestion
    pipeline can skip PDFs that have already been processed in previous runs.
    """
    with open(file_path, 'rb') as f:
        file_hash = hashlib.md5(f.read()).hexdigest()

    return file_hash


def load_processed_files():
    """
    Load the JSON log of PDFs that have already been indexed.

    Returns an empty dictionary when the log file does not exist, which means
    the pipeline is running for the first time or the log has been reset.
    """
    if os.path.exists(PROCESSED_FILES_LOG):
        with open(PROCESSED_FILES_LOG, 'r') as f:
            return json.load(f)
    return {}


def save_processed_files(processed_files):
    """
    Save the processed-file log after indexing or skipping PDFs.

    The log maps each file hash to metadata about the processed PDF, such as
    file name, source path, modification time, and number of generated nodes.
    """
    with open(PROCESSED_FILES_LOG, 'w') as f:
        json.dump(processed_files, f, indent=2)


def add_document_to_index(file_path, 
                          storage_context, 
                          processed_files, 
                          chunk_size=1024, 
                          chunk_overlap=20):
    """
    Process one PDF file and add its text nodes to the vector store.

    The function first hashes the PDF contents and checks the processed-file log.
    If the same file hash already exists, the PDF is skipped to avoid indexing it
    again. Otherwise, the PDF is loaded, split into text nodes, embedded, and
    written to Chroma through the provided LlamaIndex StorageContext.

    After successful indexing, the processed-file log is updated with basic
    metadata about the source file and generated nodes.

    Returns:
        A tuple of two booleans: (added, skipped).
        - added is True when the PDF was successfully indexed.
        - skipped is True when the PDF was already present in the log.
        - both are False when processing failed.
    """

    file_name = os.path.basename(file_path)
    file_hash = get_file_hash(file_path)

    # Use the file content hash to detect PDFs already indexed in previous runs.
    if file_hash in processed_files:
        print(f"Skipping {file_name} - already processed previously")
        return False, True  # Not added, but not a failure

    try:
        nodes = process_pdf(file_path, chunk_size, chunk_overlap)
         # Building the index embeds these nodes and writes them to Chroma.
        index = VectorStoreIndex(nodes, storage_context=storage_context)

        # Record the file only after indexing succeeds.
        processed_files[file_hash] = {
            "file_name": file_name,
            "file_path": file_path,
            "date_added": str(Path(file_path).stat().st_mtime),
            "node_count": len(nodes)
        }
        print(f"Added document to vector store: {file_name}")
        return True, False  # Successfully added, not a failure
 
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False, False  # Not added, and it's a failure  
    

def process_folder(folder_path, 
                   storage_context, 
                   processed_files, 
                   chunk_size=1024, 
                   chunk_overlap=20):
    """
    Process all PDF files in the knowledge base folder.

    The function scans the folder for files ending in .pdf, sends each PDF to
    add_document_to_index(), and tracks how many files were added, skipped, or
    failed. The processed-file log is saved after each file so progress is not
    lost if the script stops during a long ingestion run.

    Args:
        folder_path: Path to the folder containing PDF files.
        storage_context: LlamaIndex StorageContext connected to Chroma.
        processed_files: Dictionary loaded from the processed-file JSON log.
        chunk_size: Maximum size of each generated text node.
        chunk_overlap: Number of overlapping tokens between adjacent nodes.
    """
    added_count = 0
    skipped_count = 0
    failed_count = 0
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.pdf'):
            file_path = os.path.join(folder_path, filename)
            print(f"\nProcessing: {filename}")
            added, skipped = add_document_to_index(file_path, storage_context, processed_files, chunk_size, chunk_overlap)
            
            if added:
                added_count += 1
            elif skipped:
                skipped_count += 1
            else:
                failed_count += 1
            
            # Save progress after each file
            save_processed_files(processed_files)
    
    print(f"\nProcessing complete.")
    print(f"Added: {added_count} documents")
    print(f"Skipped (already processed): {skipped_count} documents")
    print(f"Failed: {failed_count} documents")
    print(f"Vector store location: {DATABASE_PATH}")
    

'''
Part 3. Run the ingestion pipeline

This section connects to the persistent Chroma vector store, loads the
processed-file log, and indexes any new PDF files found in the knowledge base
folder.
'''

if __name__ == "__main__":
    configure_embedding_model()

    # Connect LlamaIndex to the persistent Chroma vector store.
    storage_context = initialize_vector_store(DATABASE_PATH)

    # Load the record of PDFs that have already been indexed.
    processed_files = load_processed_files()

    # Process every new PDF in the knowledge base folder.
    process_folder(
        FOLDER_PATH,
        storage_context,
        processed_files,
        chunk_size=1024,
        chunk_overlap=20,
    )
