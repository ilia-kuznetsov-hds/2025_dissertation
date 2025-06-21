import os
from llama_index.readers.file import PyMuPDFReader
from llama_index.core.schema import Document
from nltk.tokenize import sent_tokenize
import re
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, SparseVectorParams
from llama_index.core import VectorStoreIndex, StorageContext
import hashlib

# Configure global embedding model
Settings.embed_model = HuggingFaceEmbedding(
    model_name='sentence-transformers/all-mpnet-base-v2')


'''
Step 1. Load PDFs and parse them into semantic chunks.
'''

def load_pdf(file_path):
    """
    Load PDF file and return Document objects
    """
    loader = PyMuPDFReader()
    document = loader.load(file_path=file_path)
    print(f"Loaded {len(document)} document(s) from {file_path}")
    return document


def create_semantic_nodes(filepath):
    '''
    Create semantic nodes from a PDF file.
    This function loads a PDF, cleans the text by removing specific patterns,
    and splits the text into semantic chunks using a specified embedding model.
    Args:
        filepath (str): Path to the PDF file.
    Returns:
        List[Document]: A list of Document objects containing cleaned and split text.
        '''
    document = load_pdf(filepath)
    file_name = os.path.basename(filepath)

    remove_patterns = [
    r"This PDF of the BMJ Best Practice topic is based on the web version that was last updated: .+\.",
    r"BMJ Best Practice topics are regularly updated and the most recent version of the topics\s*can be found on bestpractice\.bmj\.com",
    r"Use of this content is subject to our",
    r"© BMJ Publishing Group Ltd 2025\.",
    r"All rights reserved\."
    ]

    def should_remove(sentence):
        '''
        Function is used by create_semantic_chunks funtion
        Check if a sentence should be removed based on predefined patterns.
        '''
        for pattern in remove_patterns:
            if re.search(pattern, sentence.strip()):
                return True
        return False

    document_clean = []

    for i, page in enumerate(document):
        print(f"Document {file_name} page {i+1}: {page.text[:50]}...") 
        text = page.text.replace('\n', ' ').strip()
        sentences = sent_tokenize(text)
        filtered = [s for s in sentences if not should_remove(s)]
        citation_pattern = re.compile(r'\[\d+\]')
        filtered_sentences = [citation_pattern.sub('', sent).strip() for sent in filtered] 
        min_length = 4
        filtered = [s for s in filtered_sentences if len(s.strip()) >= min_length]
        metadata = {
            "source": file_name,
            "page": i
        }
        cleaned_text = " ".join(filtered)
        doc = Document(text=cleaned_text, metadata=metadata)
        document_clean.append(doc)

    embed_model = HuggingFaceEmbedding(model_name='sentence-transformers/all-mpnet-base-v2')#"sentence-transformers/all-MiniLM-L6-v2")
    splitter = SemanticSplitterNodeParser(buffer_size=3, 
                                          breakpoint_percentile_threshold=90, 
                                          embed_model=embed_model)

    nodes = splitter.get_nodes_from_documents(document_clean)

    return nodes
    

def split_long_nodes(nodes, max_length=1100):
    '''
    Split long nodes into smaller chunks based on sentence boundaries. Because semantic nodes can 
    be very long, this function ensures that each chunk does not exceed the specified maximum length.
    Args:   
        nodes (List[Document]): List of Document objects to be split.
        max_length (int): Maximum length of each chunk.
        Returns:
        List[Document]: A list of Document objects with split text.
        '''
    new_nodes = []
    for node in nodes:
        text = node.text
        if len(text) <= max_length:
            new_nodes.append(node)
        else:
            # Split by sentences for better coherence
            sentences = sent_tokenize(text)
            chunk = ""
            for sent in sentences:
                if len(chunk) + len(sent) + 1 > max_length:
                    if chunk:
                        new_nodes.append(Document(text=chunk.strip(), metadata=node.metadata))
                    chunk = sent
                else:
                    chunk += " " + sent
            if chunk:
                new_nodes.append(Document(text=chunk.strip(), metadata=node.metadata))


    def create_document_id(node):
        """Create a unique ID based on document content and metadata"""
        content = f"{node.text}_{node.metadata.get('source', '')}_{node.metadata.get('page', 0)}"
        return hashlib.md5(content.encode()).hexdigest()
    for node in new_nodes:
        node.node_id = create_document_id(node)
    
    return new_nodes


'''Step 2. Initialize Qdrant vector store and create collection.'''
def initialize_vector_store():
    '''
    Initialize Qdrant vector store and create collection if it doesn't exist.
    Returns:
        QdrantVectorStore: Initialized Qdrant vector store.
        StorageContext: Storage context for the vector store.
        '''
    client = QdrantClient(url="http://localhost:6333")
    collection_name = "qdrant_medical_documents_3"
    # Check if collection exists, if not create it
    try:
        client.get_collection(collection_name)
        print(f"Collection '{collection_name}' already exists")
    except:
        # Create collection with proper vector configuration
        '''
        For hybrid search, we need to create a collection with both dense and sparse vectors.
        Dense vectors are used for semantic search, while sparse vectors (e.g., BM25) are used for keyword search.
        We create a dict with vector names and their configurations.
        If not defined in a way that Qdrant expects, it will raise an error like this:
            raise UnexpectedResponse.for_response(response)
            qdrant_client.http.exceptions.UnexpectedResponse: Unexpected Response: 400 (Bad Request)
            Raw response content:
            b'{"status":{"error":"Wrong input: Not existing vector name error: text-sparse-new"},"time":0.006969898}'
        '''
        client.create_collection(
            collection_name=collection_name,
            vectors_config={"text-dense": VectorParams( # For hybrid search named vectors required
                size=768,  # all-mpnet-base-v2 embedding dimension is 768
                distance=Distance.COSINE
            )
            },
            sparse_vectors_config={
                "text-sparse": SparseVectorParams()
            }
        )
        print(f"Created hybrid collection '{collection_name}'")

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        fastembed_sparse_model='Qdrant/bm25',
        enable_hybrid=True,
        batch_size=20  
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    return vector_store, storage_context



def populate_index(nodes, storage_context):
    '''
    Populate the Qdrant vector store with semantic nodes.
    Args:
        nodes (List[Document]): List of Document objects to be added to the vector store.
        storage_context (StorageContext): Storage context for the vector store.
    '''
    client = QdrantClient(url="http://localhost:6333")
    collection_name = "qdrant_medical_documents_3"

    existing_ids = set()
    try:
        scroll_result = client.scroll(collection_name=collection_name, limit=10000)
        existing_ids = {point.id for point in scroll_result[0]}
        print(f"Found {len(existing_ids)} existing documents")
    except Exception as e:
        print(f"No existing documents found: {e}")

    # Filter out nodes that already exist
    new_nodes = [node for node in nodes if node.node_id not in existing_ids]
    print(f"Adding {len(new_nodes)} new documents (skipping {len(nodes) - len(new_nodes)} duplicates)")

    # This will automatically embed the nodes using the configured embedding model
    # and store them in Qdrant
    print(f"Adding {len(nodes)} nodes to Qdrant...")
    index = VectorStoreIndex(
        nodes=new_nodes,
        storage_context=storage_context,
        embed_model=Settings.embed_model
    )

    print("Successfully populated Qdrant vector store!")
    collection_info = client.get_collection(collection_name)
    print(f"Collection info: {collection_info}")

# Main execution

filepath = r"C:\\Users\\kuzne\\Desktop\\03 University\\Dissertation\\06 Medical Text Corpus\\BMJ\\Stripped PDFs\\Alzheimer's disease stripped.pdf"

long_nodes = create_semantic_nodes(filepath)

nodes = split_long_nodes(long_nodes, max_length=1500)

vector_store, storage_context = initialize_vector_store()

populate_index(nodes, storage_context)