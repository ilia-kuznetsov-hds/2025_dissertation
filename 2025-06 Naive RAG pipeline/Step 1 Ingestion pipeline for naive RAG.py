import os
from llama_index.readers.file import PyMuPDFReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext
import chromadb
import hashlib
from pathlib import Path
import json

# Folder path to the directory with PDF files of medical texts' corpus
FOLDER_PATH = r"C:\\Users\\kuzne\\Desktop\\03 University\\Dissertation\\06 Medical Text Corpus\\BMJ"
# Path to the Chroma database that stores all the embeddings
DATABASE_PATH = r"C:\\Users\\kuzne\\Documents\\Python_repo\\2025_01_dissertation\\2025_dissertation\\chromadb"
# Path to store the record of processed files
PROCESSED_FILES_LOG = r"C:\\Users\\kuzne\\Desktop\\03 University\\Dissertation\\06 Medical Text Corpus\\BMJ\\processed_files.json"

'''
Part I
PDF processing pipeline that extracts and segment text from article.
This code: 
1) Imports necessary libraries: It uses PyMuPDFReader to read PDF files and 
SentenceSplitter to break text into manageable segments.

2) Defines three functions:
* load_pdf(): Takes a file path, loads the PDF using PyMuPDFReader, and returns documents object
* parse_nodes(): Takes documents object and split it into smaller text chunks (nodes) using SentenceSplitter, 
with configurable chunk size and overlap
* process_pdf(): Orchestrates the workflow by calling the other functions and displaying preview information about 
the first 3 text nodes

'''


def load_pdf(file_path):
    '''
    The fucntion to parse a single PDF file.
    Returns Document object of LlamaIndex framework.
    Later on, this object will be used to create a list of Nodes objects 
    (basically chunks of text in the LlamaIndex framework).
    '''
    # https://pymupdf.readthedocs.io/en/latest/rag.html
    loader = PyMuPDFReader()
    documents = loader.load(file_path=file_path)
    print(f"Loaded {len(documents)} document(s) from {file_path}")
    return documents


def parse_nodes(documents, chunk_size=1024, chunk_overlap=20):
    '''
    The function to parse Document object into Nodes.
    The SentenceSplitter attempts to split text while respecting the boundaries of sentences.
    https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/modules/
    
    '''
    text_splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap)
    nodes = text_splitter.get_nodes_from_documents(documents)
    print(f"Created {len(nodes)} from the document(s)")
    return nodes


def process_pdf(file_path, chunk_size, chunk_overlap):
    '''
    Returns a list of of TextNodes
    https://docs.llamaindex.ai/en/v0.10.19/api/llama_index.core.schema.TextNode.html 

    '''
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
Part 2. Add nodes to vector store

Vector stores accept a list of Node objects and build an index from them.
LlamaIndex supports dozens of vector stores. You can specify which one to use by passing in a 
StorageContext, on which in turn you specify the vector_store argument.
In that script, we use ChromaVectorStore.
We provide it with database path to peristent database object stored at the local drive and with 
list of TextNodes that were created by process_pdf function from Step 1. 

'''

def initialize_vector_store(database_path):
    """
    Initialize the vector store and return the collection and index
    """
    client = chromadb.PersistentClient(path=database_path)
    collection = client.get_or_create_collection(name="articles")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    # If you have already computed embeddings and dumped them into an external vector store
    # https://docs.llamaindex.ai/en/stable/module_guides/indexing/vector_store_guide/
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    return vector_store, storage_context, index 


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
                          processed_files, 
                          chunk_size=1024, 
                          chunk_overlap=20):
    """
    Process a single PDF and add it to the index
    """
    file_name = os.path.basename(file_path)
    file_hash = get_file_hash(file_path)

    # Check if file was already processed
    if file_hash in processed_files:
        print(f"Skipping {file_name} - already processed previously")
        return False, True  # Not added, but not a failure

    try:
        nodes = process_pdf(file_path, chunk_size, chunk_overlap)
        # Create a new index with just these nodes
        index = VectorStoreIndex(nodes, storage_context=storage_context)
        # Record this file as processed
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
    Process all PDF files in a folder
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
Part 3. Run the script
'''
 
# Initialize the vector store
vector_store, storage_context, index = initialize_vector_store(DATABASE_PATH)
# Load record of previously processed files
processed_files = load_processed_files()
# Add all PDF files in the folder to the vector store
process_folder(FOLDER_PATH, storage_context, processed_files, chunk_size=1024, chunk_overlap=20)
