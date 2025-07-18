import os
from llama_index.readers.file import PyMuPDFReader
from llama_index.core.schema import Document
from nltk.tokenize import sent_tokenize
import re
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from llama_index.core import VectorStoreIndex, StorageContext
import hashlib

# Configure global embedding model
Settings.embed_model = HuggingFaceEmbedding(
    model_name='sentence-transformers/all-mpnet-base-v2') # By default, input text longer than 384 word pieces is truncated.


'''
Step 1. Load PDFs and parse them into chunks.
'''

def load_pdf(file_path):
    """
    Load PDF file and return Document objects
    """
    loader = PyMuPDFReader()
    document = loader.load(file_path=file_path)
    print(f"Loaded {len(document)} document(s) from {file_path}")
    return document


def create_nodes(filepath):
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
    r"© BMJ Publishing Group Ltd 2024\.",
    r"All rights reserved\."
    ]

    def should_remove(sentence):
        '''
        Function is used by create_nodes funtion
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

    
    splitter = SimpleNodeParser(
        chunk_size=1000,
        chunk_overlap=200
    )
    nodes = splitter.get_nodes_from_documents(document_clean)

    def create_document_id(node):
        """Create a unique ID based on document content and metadata"""
        content = f"{node.text}_{node.metadata.get('source', '')}_{node.metadata.get('page', 0)}"
        return hashlib.md5(content.encode()).hexdigest()
    for node in nodes:
        node.node_id = create_document_id(node)

    return nodes




def initialize_simple_vector_store():
    client = QdrantClient(url="http://localhost:6333")
    collection_name = "psychiatry_guidelines_simple_chunking"
    
    try:
        client.get_collection(collection_name)
        print(f"Collection '{collection_name}' already exists")
    except:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=768,  # all-mpnet-base-v2 embedding dimension is 768
                distance=Distance.COSINE)
          )
    
   
    
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        batch_size=32     
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    return vector_store, storage_context


def populate_index(nodes, storage_context):
    '''
    Populate the Qdrant vector store with nodes.
    Args:
        nodes (List[Document]): List of Document objects to be added to the vector store.
        storage_context (StorageContext): Storage context for the vector store.
    Returns:
        None because it populates the index that is stored on disk
    '''
    client = QdrantClient(url="http://localhost:6333")
    collection_name = "psychiatry_guidelines_simple_chunking"

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


def process_folder(folder_path, storage_context
                   ):
    """
    Process all PDF files in a folder
    """
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.pdf'):
            file_path = os.path.join(folder_path, filename)
            print(f"\nProcessing: {filename}")
            nodes = create_nodes(file_path)
            populate_index(nodes, storage_context)

    print(f"Processing complete.")

# Main execution
# Pay attention to collection name - you have to change it in 2 functions 

folder_path = r"C:\\Users\\kuzne\\Desktop\\03 University\\Dissertation\\06 Medical Text Corpus\\BMJ\\Stripped PDFs"

vector_store, storage_context = initialize_simple_vector_store()

process_folder(folder_path, storage_context)

