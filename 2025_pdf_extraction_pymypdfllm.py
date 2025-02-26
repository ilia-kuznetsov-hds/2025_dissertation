
# https://docs.llamaindex.ai/en/stable/examples/low_level/oss_ingestion_retrieval/

from pathlib import Path
from llama_index.readers.file import PyMuPDFReader
from llama_index.core.node_parser import SentenceSplitter



FILE_PATH = r"C:\\Users\\User\\Desktop\\test_path\\2025 Journal article Toward expert-level medical question answering with LLMs.pdf"

# https://pymupdf.readthedocs.io/en/latest/rag.html
loader = PyMuPDFReader()
documents = loader.load(file_path=FILE_PATH)


text_parser = SentenceSplitter(
    chunk_size=1024,
    # separator=" ",
)


text_chunks = []
# maintain relationship with source doc index, to help inject doc metadata in (3)
doc_idxs = []
for doc_idx, doc in enumerate(documents):
    cur_text_chunks = text_parser.split_text(doc.text)
    text_chunks.extend(cur_text_chunks)
    doc_idxs.extend([doc_idx] * len(cur_text_chunks))


from llama_index.core.schema import TextNode

nodes = []
for idx, text_chunk in enumerate(text_chunks):
    node = TextNode(
        text=text_chunk,
    )
    src_doc = documents[doc_idxs[idx]]
    node.metadata = src_doc.metadata
    nodes.append(node)