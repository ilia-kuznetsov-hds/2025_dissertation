import os
import chromadb
import ollama



DATABASE_PATH = r"C:\\Users\\User\\Documents\\Python_projects\\2025_dissertation\\chromadb"

JSON_FILE_PATH = r'C:\\Users\\User\\Documents\\Python_projects\\2025_dissertation\\articles_chunks.json'

'''
Import file
'''


import json

# Load JSON data from file
# This is function for test of small batch of data

def load_json_batch(filename, batch_size=5):
    with open(filename, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data[:batch_size]


def load_json(filename):
    with open(filename, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data


data = load_json(JSON_FILE_PATH) 


# Create/open DB client

client = chromadb.PersistentClient(path=DATABASE_PATH)
collection = client.get_or_create_collection(name="articles")


'''

About embedding model:
https://www.mixedbread.com/blog/mxbai-embed-large-v1

Reference from ollama:
https://ollama.com/blog/embedding-models


'''


def create_emdeddings(data):
   response = ollama.embeddings(model="mxbai-embed-large", prompt=data)
   return response["embedding"]


def process_file(data):
    for item in data:
        pmcid = str(item['pmcid'])
        title = str(item['title'])

        for chunk in item['chunks']:
            chunk_text = chunk['text']
            chunk_index = chunk['metadata']['chunk_index']
            chunk_id = f'{pmcid}_chunk_{chunk_index}'

            # Create embeddings
            embedding = create_emdeddings(chunk_text)

            # Add to DB
            collection.add(
                ids=[chunk_id], 
                embeddings=[embedding], 
                documents=[chunk_text], 
                metadatas=[{
                    "pmcid": pmcid,
                    "title": title,
                    "chunk_index": chunk_index,
                    'start_text': chunk['metadata']['start_text'],
                    'end_text': chunk['metadata']['end_text']
                }])
        print('Added:', pmcid)
    print('Done!')
            



process_file(data)
    








'''
class VectorDBManager:
    def __init__(self, collection_name: str, persist_directory: str):
        """
        Initialize the Vector Database Manager.
        
        Args:
            collection_name: Name of the collection to create/use
            persist_directory: Directory where ChromaDB will persist the data
        """
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(name=collection_name)
        
        # Ollama API endpoint
        self.model = "mxbai-embed-large"

'''
    



    





















'''


# store each document in a vector embedding database
for i, d in enumerate(docs):
  response = ollama.embeddings(model="mxbai-embed-large", 
                               prompt=d)
  embedding = response["embedding"]
  collection.add(
    ids=[str(i)],
    embeddings=[embedding],
    documents=[d]
  )
'''