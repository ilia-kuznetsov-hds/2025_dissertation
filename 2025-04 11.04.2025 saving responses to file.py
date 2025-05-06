import os
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.llms.together import TogetherLLM
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.llms import ChatMessage
import pandas as pd
from datetime import datetime



together_api_key = os.getenv("TOGETHER_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

DATABASE_PATH = r"C:\\Users\\kuzne\\Documents\\Python_repo\\2025_01_dissertation\\2025_dissertation\\chromadb"

QUESTIONS_FILE_PATH = r"C:\\Users\\kuzne\\Documents\\Python_repo\\2025_01_dissertation\\2025_dissertation\\data\\reviewed_questions\\ptsd_reviewed_questions test.xlsx"
OUTPUT_FILE_PATH = r"C:\\Users\\kuzne\\Documents\\Python_repo\\2025_01_dissertation\\2025_dissertation\\data\\reviewed_questions\\ptsd_reviewed_questions test_with_answers.xlsx"


'''
Part 1. Connect to external vector store (with existing embeddings)
https://docs.llamaindex.ai/en/stable/module_guides/indexing/vector_store_guide/
This is the way to access previously calculated embeddings stored in the index
'''
client = chromadb.PersistentClient(path=DATABASE_PATH)
collection = client.get_or_create_collection(name="articles")
vector_store = ChromaVectorStore(chroma_collection=collection)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)



df = pd.read_excel(QUESTIONS_FILE_PATH)
user_query = 'How do we treat bipolar disorder in adults?'


'''
Part 2. Define fucntions to generate RAG-enhanced LLM responses

'''

def generate_rag_response(question, model_name="deepseek"):
    """
    Generate a response using RAG-enhanced LLM
    
    """
    global index
    user_query = question
    query_engine = index.as_query_engine(similarity_top_k=3)
    response = query_engine.query(user_query)
    source_texts = []
    for source_node in response.source_nodes:
        source_texts.append(source_node.node.text)
    context = "\n\n".join(source_texts)
    #print(f' Retrieved context: {context}')
    
    prompt = f"""You are an expert assistant with access to specific information. 
          Use the following context to answer the user question. If the context doesn't 
          contain relevant information, state that you don't have enough information.

          CONTEXT: {context}
          USER QUESTION: {user_query}

        Your response should be comprehensive, accurate, and based on the context provided.
        You should highlight the parts of the context that are relevant to the user question and show how you used it."""
    
    if model_name.lower() == "deepseek":
        llm = TogetherLLM(
            model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
            api_base="https://api.together.xyz/v1",
            api_key=together_api_key,
            is_chat_model=True,
            is_function_calling_model=True,
            temperature=0.1
        )

        messages=[ChatMessage(role = "user", content = prompt)]
        full_response = llm.chat(messages)
        answer_text = full_response.message.content

    else:
        llm = GoogleGenAI(
            model="gemini-2.0-flash",
            api_key=google_api_key  
        )
        full_response = llm.complete(prompt)
        answer_text = full_response.text

    return answer_text  



def generate_vanilla_response(question, model_name="deepseek"):
    
    user_query = question
    prompt = f"""You are an expert assistant. Answer the user question based on your knowledge.
        If you don't have enough information, state that you don't have enough information.
          
          USER QUESTION: {user_query}

        Your response should be comprehensive and accurate."""

    if model_name.lower() == "deepseek":
        llm = TogetherLLM(model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
            api_base="https://api.together.xyz/v1",
            api_key=together_api_key,
            is_chat_model=True,
            is_function_calling_model=True,
            temperature=0.1
        )
        messages = [ChatMessage(role="user", content=prompt)]
        # chat() is the method that sends meggases to LLM and receives the reponses
        # It takes a list of messages as input and returns the full response
        full_response = llm.chat(messages)
        answer_text = full_response.message.content

    else:
        llm = GoogleGenAI(
            model="gemini-2.0-flash",
            api_key=google_api_key  
        )
        full_response = llm.complete(prompt)
        answer_text = full_response.text
    
    return answer_text


'''
Part 3. Define function to process questions from file and save results

'''

def process_questions_from_csv(file_path, row_index, model_name, output_file=None):
    df = pd.read_excel(file_path)

    if 'Generated Answer With RAG' not in df.columns:
        df['Generated Answer With RAG'] = None
        df['Model'] = None
        df['Response Date'] = None
        df['Response Time (seconds)'] = None

    # Check if row index is valid
    if row_index < 0 or row_index >= len(df):
        raise ValueError(f"Row index {row_index} is out of bounds (file has {len(df)} rows)")
    
    # Get the question
    question = df.at[row_index, 'Modified Questions']
    print(f"Processing question at row {row_index}: {question}")

    # Record start time for performance measurement
    start_time = datetime.now()
 
    answer_rag = generate_rag_response(question, model_name=model_name)
    answer_vanilla = generate_vanilla_response(question, model_name=model_name)

    # Record end time and calculate duration
    end_time = datetime.now()
    duration_seconds = (end_time - start_time).total_seconds()

    # Debug check to confirm we have the answer
    #print(f"Debug - Captured answer: {answer[:50]}...")

    df.at[row_index, 'Generated Answer With RAG'] = answer_rag
    df.at[row_index, 'Generated Vanilla Answer'] = answer_vanilla
    df.at[row_index, 'Model'] = model_name
    df.at[row_index, 'Response Date'] = end_time.strftime("%Y-%m-%d %H:%M:%S")
    df.at[row_index, 'Response Time (seconds)'] = duration_seconds

    # If no output_file specified, create one based on the input filename
    if output_file is None:
        base_name = os.path.basename(file_path)
        name_without_ext = os.path.splitext(base_name)[0]
        output_file = os.path.join(os.path.dirname(file_path), f"{name_without_ext}_with_answers {model_name}.xlsx")

    # Save the updated DataFrame
    df.to_excel(output_file, index=False)
    print(f"Results saved to {output_file}")

    return df


#process_questions_from_csv(QUESTIONS_FILE_PATH, 3, model_name="deepseek")
#process_questions_from_csv(QUESTIONS_FILE_PATH, 1, model_name="gemini")


QUESTIONS_GEMINI = r"C:\\Users\\kuzne\\Documents\\Python_repo\\2025_01_dissertation\\2025_dissertation\\data\\reviewed_questions\\ptsd_reviewed_questions test_with_answers gemini.xlsx"

process_questions_from_csv(QUESTIONS_GEMINI, 2, model_name="gemini", output_file=QUESTIONS_GEMINI)