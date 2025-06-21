from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
# Step 5: Create VectorStoreIndex and populate with nodes
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Configure global embedding model
Settings.embed_model = HuggingFaceEmbedding(
    model_name='sentence-transformers/all-mpnet-base-v2')#"sentence-transformers/all-MiniLM-L6-v2")

client = QdrantClient(url="http://localhost:6333")
# Define collection name
collection_name = "medical_documents"

# Step 4: Initialize Qdrant vector store
vector_store = QdrantVectorStore(
    client=client,
    collection_name=collection_name
)



storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Load existing index from vector store (since you mentioned it's already populated)
index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    storage_context=storage_context
)


# Optional: Test a simple query to verify everything works
query_engine = index.as_query_engine(similarity_top_k=3)
test_query = "What are the symptoms of Alzheimer's disease?"
response = query_engine.query(test_query)
print(f"\nTest query: {test_query}")
print(f"Response: {response}")