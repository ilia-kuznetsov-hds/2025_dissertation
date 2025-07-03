# https://docs.llamaindex.ai/en/stable/examples/low_level/oss_ingestion_retrieval/
from pathlib import Path
from llama_index.readers.file import PyMuPDFReader
from llama_index.core.node_parser import SentenceSplitter

import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from IPython.display import Markdown, display
from llama_index.core import StorageContext

# https://docs.llamaindex.ai/en/stable/examples/llm/together/
from llama_index.llms.together import TogetherLLM
import os


from llama_index.core import VectorStoreIndex, get_response_synthesizer
# Can be used later for customizing retriever
# https://docs.llamaindex.ai/en/stable/understanding/querying/querying/
#from llama_index.core.retrievers import VectorIndexRetriever
#from llama_index.core.query_engine import RetrieverQueryEngine
#from llama_index.core.postprocessor import SimilarityPostprocessor


from llama_index.core.llms import ChatMessage



my_api_key = os.getenv("TOGETHER_API_KEY")
# FILE_PATH = r"C:\\Users\\User\\Desktop\\test_path\\2025 Journal article Toward expert-level medical question answering with LLMs.pdf"
#FILE_PATH = r"C:\\Users\\kuzne\\Desktop\\rag_articles_pdf\\2025 Almanac - Retrieval Augmented Language models for clinical medicine.pdf"
FILE_PATH = r"C:\\Users\\kuzne\\Desktop\\03 University\\Dissertation\\06 Medical Text Corpus\\BMJ\\Bipolar disorder in adults.pdf"

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



'''
This is commented part. 
For now, we create index object and work with it using VectorStore.
Later we will create embeddings for nodes and store them in the database.

The following code generates embeddings for text from Node:

# you cannot use TextNode object directly
# https://docs.pydantic.dev/2.10/errors/validation_errors/#string_type
# When trying to use the get_text_embedding_batch() method, 
# you're passing the entire node objects, but this method expects a list of strings. 
node_texts = [node.text for node in nodes_file]

from llama_index.embeddings.ollama import OllamaEmbedding
ollama_embedding = OllamaEmbedding(
    model_name="mxbai-embed-large")

pass_embedding = ollama_embedding.get_text_embedding_batch(
    node_texts, show_progress=True
)
#print(pass_embedding)

'''


'''
Part 3. 
RAG Query with Together API Integration

'''

user_query = 'How do we screen bipolar disorder in adults?'

def generate_response_from_llm(user_query, vector_store):
    user_query = user_query
    vector_store = vector_store

    query_engine = vector_store.as_query_engine(
        similarity_top_k=5)
    response = query_engine.query(user_query)

    source_texts = []
    for source_node in response.source_nodes:
        source_texts.append(source_node.node.text)
    
    context = "\n\n".join(source_texts)
    print(f' Retrieved context: {context}')
    
    prompt = f"""You are an expert assistant with access to specific information. 
          Use the following context to answer the user question. If the context doesn't 
          contain relevant information, state that you don't have enough information.

          CONTEXT: {context}
          USER QUESTION: {user_query}

        Your response should be comprehensive, accurate, and based on the context provided."""
    
    llm = TogetherLLM(
    model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
    api_base="https://api.together.xyz/v1",
    api_key=my_api_key,
    is_chat_model=True,
    is_function_calling_model=True,
    temperature=0.1
    )
    
    print("\nAnswer: ", end="", flush=True)

    messages=[
    ChatMessage(role = "user", 
                content = prompt)]
    

    # https://docs.llamaindex.ai/en/stable/examples/llm/together/
    # pip install llama-index-llms-together
    response = llm.stream_chat(messages)
    for r in response:
        print(r.delta, end="")



generate_response_from_llm(user_query, index_vector_store)



