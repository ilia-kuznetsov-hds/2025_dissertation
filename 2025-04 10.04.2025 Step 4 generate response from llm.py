import os
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.llms.together import TogetherLLM
from llama_index.core.llms import ChatMessage
import pandas as pd


my_api_key = os.getenv("TOGETHER_API_KEY")

DATABASE_PATH = r"C:\\Users\\kuzne\\Documents\\Python_repo\\2025_01_dissertation\\2025_dissertation\\chromadb"

QUESTIONS_FILE_PATH = r"C:\\Users\\kuzne\\Documents\\Python_repo\\2025_01_dissertation\\2025_dissertation\\data\\reviewed_questions\\ptsd_reviewed_questions test.xlsx"


'''
Part 1. Connect to external vector store (with existing embeddings)
https://docs.llamaindex.ai/en/stable/module_guides/indexing/vector_store_guide/

This is the way to access previously calculated embeddings stored in the index

'''

# https://docs.llamaindex.ai/en/stable/examples/vector_stores/ChromaIndexDemo/
# Bsic example including saing to the disc
client = chromadb.PersistentClient(path=DATABASE_PATH)
collection = client.get_or_create_collection(name="articles")
vector_store = ChromaVectorStore(chroma_collection=collection)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)


df = pd.read_excel(QUESTIONS_FILE_PATH)

user_query = 'How do we treat bipolar disorder in adults?'

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
    
    # Use non-streaming version to capture the full response
    full_response = llm.chat(messages)
    answer_text = full_response.message.content
    print(answer_text)  # Print the answer
    return answer_text  # Return the answer

    



#generate_response_from_llm(user_query, index) 




def process_questions_from_csv(file_path, row_index, vector_store, output_file=None):
    df = pd.read_excel(file_path)
    df['Generated Answer'] = None  # Create a new column for generated answers

    # Check if row index is valid
    if row_index < 0 or row_index >= len(df):
        raise ValueError(f"Row index {row_index} is out of bounds (file has {len(df)} rows)")
    
    # Get the question
    question = df.at[row_index, 'Modified Questions']
    print(f"Processing question at row {row_index}: {question}")
 
    answer = generate_response_from_llm(question, vector_store)

    # Debug check to confirm we have the answer
    print(f"Debug - Captured answer: {answer[:50]}...")

    df.at[row_index, 'Generated Answer'] = answer

    # If no output_file specified, create one based on the input filename
    if output_file is None:
        base_name = os.path.basename(file_path)
        name_without_ext = os.path.splitext(base_name)[0]
        output_file = os.path.join(os.path.dirname(file_path), f"{name_without_ext}_with_answers.csv")

    # Save the updated DataFrame
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
   

 

    return df


OUTPUT_FILE_PATH = r"C:\\Users\\kuzne\\Documents\\Python_repo\\2025_01_dissertation\\2025_dissertation\\data\\reviewed_questions\\ptsd_reviewed_questions test_with_answers.csv"

process_questions_from_csv(file_path=QUESTIONS_FILE_PATH, 
                           row_index=1, 
                           vector_store=index, 
                           output_file=OUTPUT_FILE_PATH)


