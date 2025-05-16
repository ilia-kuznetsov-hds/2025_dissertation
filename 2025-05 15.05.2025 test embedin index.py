import chromadb

FILE_PATH = r"C:\\Users\\kuzne\\Desktop\\03 University\\Dissertation\\06 Medical Text Corpus\\BMJ\\Post-traumatic stress disorder.pdf"

DATABASE_PATH = r"C:\\Users\\kuzne\\Documents\\Python_repo\\2025_01_dissertation\\2025_dissertation\\chromadb_huggingfaceembed"


def check_vector_store(database_path, collection_name="articles"):
    client = chromadb.PersistentClient(path=database_path)
    collection = client.get_collection(collection_name)
    
    # Get collection info
    collection_info = collection.count()
    print(f"Collection contains {collection_info} documents")
    
    # Get a sample of embeddings (first 5)
    if collection_info > 0:


        # Get all embeddings from the collection
        result = collection.get(include=['embeddings', 'documents', 'metadatas'])
        
        embeddings = result['embeddings']
        documents = result['documents']
        metadatas = result['metadatas']

        print(f"First 5 embeddings: {embeddings[:5]}")
    
        
    

# Check if embeddings exist after creating the index
has_embeddings = check_vector_store(DATABASE_PATH)