import os
from llama_index.readers.file import PyMuPDFReader
from llama_index.core.schema import Document
from nltk.tokenize import sent_tokenize
import re
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client import QdrantClient
import hashlib


# Configure global embedding model
Settings.embed_model = HuggingFaceEmbedding(
    model_name='sentence-transformers/all-mpnet-base-v2')#"sentence-transformers/all-MiniLM-L6-v2")

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
                                          breakpoint_percentile_threshold=80, 
                                          embed_model=embed_model)

    nodes = splitter.get_nodes_from_documents(document_clean)

    return nodes
    


def split_long_nodes(nodes, max_length=1100):
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


filepath = r"C:\\Users\\kuzne\\Desktop\\03 University\\Dissertation\\06 Medical Text Corpus\\BMJ\\Stripped PDFs\\Alzheimer's disease stripped.pdf"

long_nodes = create_semantic_nodes(filepath)

nodes = split_long_nodes(long_nodes, max_length=1500)


client = QdrantClient(url="http://localhost:6333")
# Define collection name
collection_name = "medical_documents"


# Step 3: Create collection if it doesn't exist
from qdrant_client.models import Distance, VectorParams

# Check if collection exists, if not create it
try:
    client.get_collection(collection_name)
    print(f"Collection '{collection_name}' already exists")
except:
    # Create collection with proper vector configuration
    # all-mpnet-base-v2 produces 768-dimensional vectors
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=768,  # all-mpnet-base-v2 embedding dimension
            distance=Distance.COSINE
        )
    )
    print(f"Created collection '{collection_name}'")

# Step 4: Initialize Qdrant vector store
vector_store = QdrantVectorStore(
    client=client,
    collection_name=collection_name
)

# Step 5: Create VectorStoreIndex and populate with nodes
from llama_index.core import VectorStoreIndex, StorageContext

storage_context = StorageContext.from_defaults(vector_store=vector_store)



def populate_index(nodes, storage_context):

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

    # Step 6: Verify the data was stored
    collection_info = client.get_collection(collection_name)
    print(f"Collection info: {collection_info}")


populate_index(nodes, storage_context)