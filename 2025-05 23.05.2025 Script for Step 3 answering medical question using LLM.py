import os
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.llms.together import TogetherLLM
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.llms.groq import Groq
from llama_index.core.llms import ChatMessage
import pandas as pd
import time

# Providers API Keys
together_api_key = os.getenv("TOGETHER_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Path to ChromaDB persistent client
DATABASE_PATH = r"C:\\Users\\kuzne\\Documents\\Python_repo\\2025_01_dissertation\\2025_dissertation\\chromadb"
QUESTIONS_FILE_PATH = r"C:\\Users\\kuzne\\Documents\\Python_repo\\2025_01_dissertation\\2025_dissertation\\data\\2025-05 22.05.25 dataset for evaluation\\psychiatry_train_dataset.csv"

# Define available models for each provider
provider_models = {
    "deepseek": ["deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free", #https://api.together.ai/models/deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free
                 "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"],  #https://api.together.ai/models/meta-llama/Llama-3.3-70B-Instruct-Turbo-Free

    "groq": ["llama-3.3-70b-versatile", # https://console.groq.com/docs/models
             "llama-3-8b-8192",
             "gemma2-9b-it",
             "allam-2-7b"],
    "gemini": ["gemini-2.0-flash"]
}

'''
Part 1. Connect to external vector store (with existing embeddings)
https://docs.llamaindex.ai/en/stable/module_guides/indexing/vector_store_guide/

This is the way to access previously calculated embeddings stored in the index
https://docs.llamaindex.ai/en/stable/examples/vector_stores/ChromaIndexDemo/
#Basic example including saving to the disc
'''

def initialize_vector_store(database_path):
    """
    Initialize the vector store that contains precalculated emdeddings from medical text corpus.
    Returns: the vector store, storage context, index
    Collection name: "articles" - naive RAG, ingestion pipeline without preprocessing of PDF files.
    """
    client = chromadb.PersistentClient(path=database_path)
    collection = client.get_or_create_collection(name="articles")
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    # If you have already computed embeddings and dumped them into an external vector store
    # https://docs.llamaindex.ai/en/stable/module_guides/indexing/vector_store_guide/
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    return vector_store, storage_context, index 


def generate_rag_response(user_query, 
                          provider_name="deepseek", 
                          model_name="DeepSeek-R1-Distill-Llama-70B-free"):
    """
    Generate a response using RAG-enhanced LLM
    
    Returns:
    - answer_text: The generated answer text from LLM provided with context
    - context: The context retrieved from the vector store and used to generate the answer
    """
    global index
    user_query = user_query
    query_engine = index.as_query_engine(similarity_top_k=3)
    response = query_engine.query(user_query)
    source_texts = []
    for source_node in response.source_nodes:
        source_texts.append(source_node.node.text)
    
    context = "\n\n".join(source_texts)
    
    prompt = f"""
        You are an expert assistant with access to specific information. 
        Use the following context to answer the user question. If the context doesn't 
        contain relevant information, state that you don't have enough information.
        CONTEXT: {context}
        USER QUESTION: {user_query}
        Your response should be comprehensive, accurate, and based on the context provided.
        """
    
    if provider_name.lower() == "deepseek":
        llm = TogetherLLM(
            model=model_name,
            api_base="https://api.together.xyz/v1",
            api_key=together_api_key,
            is_chat_model=True,
            is_function_calling_model=True,
            temperature=0.1
        )

        messages=[ChatMessage(role = "user", content = prompt)]
        # Use non-streaming version to capture the full response
        full_response = llm.chat(messages)
        answer_text = full_response.message.content

    elif provider_name.lower() == "groq":
        llm = Groq(model=model_name, 
                   api_key=groq_api_key)
        full_response = llm.complete(prompt)
        answer_text = full_response

    else:
        llm = GoogleGenAI(
            model=model_name,
            api_key=google_api_key  
        )
        full_response = llm.complete(prompt)
        answer_text = full_response.text

    return answer_text, context


def generate_vanilla_response(user_query, 
                              provider_name="deepseek", 
                              model_name="DeepSeek-R1-Distill-Llama-70B-free"):
    """
    Generate a response using LLM without RAG
    Returns:
    - answer_text: The generated answer text from LLM
    """
    
    user_query = user_query
    prompt = f"""
        You are an expert assistant. Answer the user question based on your knowledge.
        If you don't have enough information, state that you don't have enough information.
        USER QUESTION: {user_query}
        Your response should be comprehensive and accurate.
        """

    if provider_name.lower() == "deepseek":
        llm = TogetherLLM(
            model=model_name,
            api_base="https://api.together.xyz/v1",
            api_key=together_api_key,
            is_chat_model=True,
            is_function_calling_model=True,
            temperature=0.1
        )
        messages = [ChatMessage(role="user", content=prompt)]
        # chat() is the method that sends messages to LLM and receives the reponses
        # It takes a list of messages as input and returns the full response
        full_response = llm.chat(messages)
        answer_text = full_response.message.content

    elif provider_name.lower() == "groq":
        llm = Groq(model=model_name, 
                   api_key=groq_api_key)
        full_response = llm.complete(prompt)
        answer_text = full_response

    else:
        llm = GoogleGenAI(
            model=model_name,
            api_key=google_api_key  
        )
        full_response = llm.complete(prompt)
        answer_text = full_response.text
    
    return answer_text


def process_questions_from_csv(file_path, 
                               provider_name,
                               model_name,
                               batch_size: int, 
                               max_rows: int,  
                               timeout_interval = 1,
                               timeout_seconds = 10):
    """
    Process questions from a CSV file and generate answers using LLM.
    Parameters:
    - file_path: Path to the CSV file containing questions
    - provider_name: Name of the LLM provider (e.g., "deepseek", "groq", "gemini")
    - model_name: Name of the LLM model to use. Name of the provider and model should be 
        in the provider_models dictionary (defined at the beginning of the script with other constants)
    - batch_size: Number of questions to process before saving progress in to the file
    - max_rows: Maximum number of rows to process in one run (adjusted for LLM rate limits)
    - timeout_interval: Number of questions to process before taking a break (adjusted for LLM rate limits)
    - timeout_seconds: Duration of the break in seconds (adjusted for LLM rate limits)

    Returns:
    - None
        The function saves the generated answers to a new CSV file. The name of the file is composed of the original file name,
        the provider name, and the model name. The file is saved in the same directory as the original file.

    Fucntionality:
    - The function reads the CSV file and checks if there is a file with partially questions. If such file exists, it resumes from that file.
      If not, it updates the dataframe with new columns for generated answers and metrics for future evaluation.
    - If there are unanswered questions, it generates answers using the specified LLM provider and model.
    - The function updates and saves CSV file after processing each batch of questions.
    
    """

    FILE_PATH = file_path
    BATCH_SIZE = batch_size
    PROVIDER = provider_name
    MODEL = model_name

    # Clean model name to use in filename by replacing slashes with underscores
    # Together AI models contain slach in their names
    safe_model_name = model_name.replace('/', '_')
    base_name = os.path.basename(FILE_PATH)
    file_name_without_ext = os.path.splitext(base_name)[0]
    OUTPUT_PATH = os.path.join(os.path.dirname(FILE_PATH), 
                              f"{file_name_without_ext}_{provider_name}_{safe_model_name}_answered.csv")

    # Check if the output file already exists 
    if os.path.exists(OUTPUT_PATH):
        print(f"Resuming from existing file: {OUTPUT_PATH}")
        df = pd.read_csv(OUTPUT_PATH)
    else:
        print(f"Starting new processing on: {FILE_PATH}")
        df = pd.read_csv(FILE_PATH)
        df['Model'] = MODEL
        df['Provider'] = PROVIDER
        df['Generated Vanilla Answer'] = None
        df['Semanthic Similarity for vanilla'] = None
        df['Answer Correctness for vanilla'] = None
        df['Generated RAG Answer'] = None
        df['Retrieved Context'] = None
        df['Semanthic Similarity for RAG'] = None
        df['Answer Correctness for RAG'] = None

    # Get rows of unanswered questions
    # The dataframe is updated in a way that if vanilla answer is missing, it means that RAG response is also missing.
    # So we check only for vanilla response. 
    rows_to_process = df[df['Generated Vanilla Answer'].isna()].index.tolist()
    # Limit to max_rows
    rows_to_process = rows_to_process[:max_rows]
    total_rows = len(rows_to_process)    
    print(f"Found {len(df[df['Generated Vanilla Answer'].isna()])} unanswered questions total")
    print(f"Will process {total_rows} questions in this run (max_rows={max_rows})")

    for i, idx in enumerate(rows_to_process):
        user_query = df.loc[idx, 'Modified Questions']
         # Skip if empty question
        if pd.isna(user_query) or str(user_query).strip() == '':
            df.loc[idx, 'Generated Vanilla Answer'] = "invalid"
            # Skip to the next iteration
            print(f"Skipping empty question at row {idx}")
            continue
        
        # Generate non-RAG response
        vanilla_response = generate_vanilla_response(user_query,
                                                    provider_name=PROVIDER, 
                                                    model_name=MODEL)
        # Add a timeout between vanilla and RAG response generation to avoid rate limits
        time.sleep(timeout_seconds)
            
        # Handle API errors
        if vanilla_response is None:
            print(f"Error encountered. Saving progress and exiting.")
            df.to_csv(OUTPUT_PATH, index=False)
            return
        
        rag_response, context = generate_rag_response(user_query,
                                                provider_name=PROVIDER, 
                                                model_name=MODEL)
      
         # Handle API errors
        if rag_response is None:
            print(f"Error encountered. Saving progress and exiting.")
            df.to_csv(OUTPUT_PATH, index=False)
            return

        # Update the dataframe
        df.loc[idx, 'Generated Vanilla Answer'] = vanilla_response
        df.loc[idx, 'Generated RAG Answer'] = rag_response
        df.loc[idx, 'Retrieved Context'] = context
        print(f"Processed {i+1}/{total_rows}: Row {idx} - Vanilla Answer: {vanilla_response[:50]}...")

        # Save after each batch
        if (i + 1) % BATCH_SIZE == 0:
            print(f"Saving progress after batch...")
            df.to_csv(OUTPUT_PATH, index=False)
        
        # Add timeout after specified interval
        if (i + 1) % timeout_interval == 0 and i+1 < total_rows:
            print(f"Taking a {timeout_seconds} second break to avoid rate limits...")
            time.sleep(timeout_seconds)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Results saved to {OUTPUT_PATH}")
    return None


'''
Main function
'''

# Initialize the vector store
vector_store, storage_context, index = initialize_vector_store(DATABASE_PATH)

# Test run for Gemini model
'''
process_questions_from_csv(QUESTIONS_FILE_PATH, 
                           provider_name='gemini',
                           model_name='gemini-2.0-flash',
                           batch_size = 1, 
                           max_rows=1)
'''


# Test ru for Together AI
process_questions_from_csv(QUESTIONS_FILE_PATH,
                           provider_name='deepseek',
                           model_name='deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free',
                           batch_size = 1, 
                           max_rows=10)
