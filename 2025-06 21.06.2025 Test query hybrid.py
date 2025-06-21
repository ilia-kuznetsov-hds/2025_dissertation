from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client import QdrantClient
from llama_index.core import VectorStoreIndex


# Configure global embedding model
Settings.embed_model = HuggingFaceEmbedding(
    model_name='sentence-transformers/all-mpnet-base-v2')


def activate_query_engine(similarity_top_k=2, sparse_top_k=12):
    """
    Activate Qdrant for hybrid search.
    This function initializes the Qdrant client and vector store,
    and sets up the query engine for hybrid search.
    """
    # Initialize Qdrant client
    client = QdrantClient(url="http://localhost:6333")
    
    # Define collection name
    collection_name = "qdrant_medical_documents_3"
    
    # Check if the collection exists
    try:
        client.get_collection(collection_name)
    except Exception as e:
        print(f"Collection {collection_name} does not exist: {e}")
        return None
    
    # Initialize QdrantVectorStore
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        fastembed_sparse_model='Qdrant/bm25',
        enable_hybrid=True
    )
    
    # Create index from vector store
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=Settings.embed_model
    )
    
    # Create query engine with hybrid search capabilities
    query_engine = index.as_query_engine(
        similarity_top_k=similarity_top_k, 
        sparse_top_k=sparse_top_k, 
        vector_store_query_mode="hybrid"
    )
    
    return query_engine


query_engine = activate_query_engine(
    similarity_top_k=2, 
    sparse_top_k=12
)


response = query_engine.query(
    "What are symptoms of Alzheimer's disease?"
)

print(response)