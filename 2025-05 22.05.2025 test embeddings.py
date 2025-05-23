import os
from llama_index.readers.file import PyMuPDFReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext
import chromadb



DATABASE_PATH = r"C:\\Users\\kuzne\\Documents\\Python_repo\\2025_01_dissertation\\2025_dissertation\\chromadb"

def initialize_vector_store(database_path):
    """Initialize the vector store and return the collection and index"""
    client = chromadb.PersistentClient(path=database_path)
    collection = client.get_or_create_collection(name="articles")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    # If you have already computed embeddings and dumped them into an external vector store
    # https://docs.llamaindex.ai/en/stable/module_guides/indexing/vector_store_guide/
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    return vector_store, storage_context, index 

# Initialize the vector store
vector_store, storage_context, index = initialize_vector_store(DATABASE_PATH)


def get_embeddings_from_chroma(database_path):
    client = chromadb.PersistentClient(path=database_path)
    collection = client.get_or_create_collection(name="articles")
    
    # Get all embeddings from the collection
    result = collection.get(include=['embeddings', 'documents', 'metadatas'])
    
    embeddings = result['embeddings']
    documents = result['documents']
    metadatas = result['metadatas']
    
    return embeddings, documents, metadatas

# Example usage
embeddings, documents, metadatas = get_embeddings_from_chroma(DATABASE_PATH)
print(f"Retrieved {len(embeddings)} embeddings")
print(f"First embedding shape: {len(embeddings[0])} dimensions")

print(len(documents))