import os
from llama_index.readers.file import PyMuPDFReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext
import chromadb

FILE_PATH = r"C:\\Users\\kuzne\\Desktop\\03 University\\Dissertation\\06 Medical Text Corpus\\BMJ\\Post-traumatic stress disorder.pdf"

DATABASE_PATH = r"C:\\Users\\kuzne\\Documents\\Python_repo\\2025_01_dissertation\\2025_dissertation\\chromadb"


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
    # https://pymupdf.readthedocs.io/en/latest/rag.html
    loader = PyMuPDFReader()
    documents = loader.load(file_path=file_path)
    print(f"Loaded {len(documents)} document(s) from {file_path}")
    return documents


def parse_nodes(documents, chunk_size=1024, chunk_overlap=20):
    # The SentenceSplitter attempts to split text while respecting the boundaries of sentences.
    # https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/modules/
    text_splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap)
    nodes = text_splitter.get_nodes_from_documents(documents)
    print(f"Created {len(nodes)} from the document(s)")
    return nodes


def process_pdf(file_path, chunk_size, chunk_overlap):
    documents = load_pdf(file_path)
    nodes = parse_nodes(documents, chunk_size, chunk_overlap)
    for i, node in enumerate(nodes[:3]):
        print(f'Node {i}')
        print(f' {len(node.text)}')
        print(f' First 100 chars: {node.text[:100]}')
    if len(nodes) > 3:
        print(f"... and {len(nodes) - 3} more nodes")
    
    return nodes


# List of TextNodes
# https://docs.llamaindex.ai/en/v0.10.19/api/llama_index.core.schema.TextNode.html
nodes_file = process_pdf(file_path=FILE_PATH, chunk_size=1024, chunk_overlap=20)


'''
Part 2. Add nodes to vector store

Vector stores accept a list of Node objects and build an index from them.
LlamaIndex supports dozens of vector stores. You can specify which one to use by passing in a 
StorageContext, on which in turn you specify the vector_store argument.
In that script, we use ChromaVectorStore.
We provide it with database path to peristent database object stored at the local drive and with 
list of TextNodes that were created by process_pdf function from Step 1. 

'''


def initiate_vector_store(database_path, nodes):
    client = chromadb.PersistentClient(path=database_path)
    collection = client.get_or_create_collection(name="articles")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes, storage_context=storage_context)
    return index 

# index object 
index_vector_store = initiate_vector_store(DATABASE_PATH, nodes_file)

