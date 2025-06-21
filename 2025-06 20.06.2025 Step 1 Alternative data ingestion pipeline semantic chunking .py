import os
from llama_index.readers.file import PyMuPDFReader
from llama_index.core.schema import Document
from nltk.tokenize import sent_tokenize
import re
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

# Configure global embedding model
Settings.embed_model = HuggingFaceEmbedding(
    model_name='sentence-transformers/all-mpnet-base-v2')#"sentence-transformers/all-MiniLM-L6-v2")

# Path to the alternative Chroma database 
DATABASE_PATH = r"C:\\Users\\kuzne\\Documents\\Python_repo\\2025_01_dissertation\\2025_dissertation\\chromadb_semanthic_chunking"

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
    
filepath = r"C:\\Users\\kuzne\\Desktop\\03 University\\Dissertation\\06 Medical Text Corpus\\BMJ\\Stripped PDFs\\Alzheimer's disease stripped.pdf"

nodes = create_semantic_nodes(filepath)

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
    return new_nodes

# ...existing code...


nodes = split_long_nodes(nodes, max_length=1500)

for i, node in enumerate(nodes):
    print(f"Node {i}: {node.text}...")