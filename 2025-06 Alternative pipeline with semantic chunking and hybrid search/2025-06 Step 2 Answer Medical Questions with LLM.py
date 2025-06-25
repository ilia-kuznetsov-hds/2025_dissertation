from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client import QdrantClient
from llama_index.core import VectorStoreIndex
import os
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.llms.groq import Groq
from llama_index.llms.together import TogetherLLM
from llama_index.core.llms import ChatMessage
import pandas as pd
import time

# Configure global embedding model
# You need too do it, because by default LlmaIndex uses OpenAI embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name='sentence-transformers/all-mpnet-base-v2')

# By default, response senthesizer is set to OpenAI, but we want to use Google GenAI
google_api_key = os.getenv("GOOGLE_API_KEY")
Settings.llm = GoogleGenAI(model="gemini-2.0-flash")

groq_api_key = os.getenv("GROQ_API_KEY")
together_api_key = os.getenv("TOGETHER_API_KEY")


def activate_query_engine(similarity_top_k=2, sparse_top_k=12, llm = None):
    """
    Activate Qdrant for hybrid search.
    This function initializes the Qdrant client populated with psychiatry guidelines
    and sets up the query engine for hybrid search.
    Args:
        similarity_top_k (int): Number of top similar vectors to return.
        sparse_top_k (int): Number of top sparse vectors to return.
    Returns:
        query_engine (QueryEngine): A query engine configured for hybrid search.
    """
    # Initialize Qdrant client
    client = QdrantClient(url="http://localhost:6333")
    
    # Define collection name
    # The collection name for last pipeline - "psychiatry_guidelines"
    collection_name = "psychiatry_guidelines"
    
    # Check if the collection exists
    try:
        client.get_collection(collection_name)
    except Exception as e:
        print(f"Collection {collection_name} does not exist: {e}")
        return None
    
    # Initialize QdrantVectorStore
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        fastembed_sparse_model='Qdrant/bm25',
        enable_hybrid=True
    )
    
    # Create index from vector store
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=Settings.embed_model
    )
    
    # Create query engine with hybrid search capabilities
    query_engine = index.as_query_engine(
        similarity_top_k=similarity_top_k, 
        sparse_top_k=sparse_top_k, 
        vector_store_query_mode="hybrid",
        llm=llm  # Use provided LLM or default LLM
    )
    
    return query_engine


def generate_rag_response(user_query: str,  
                          similarity_top_k=3, 
                          sparse_top_k=10,
                          provider_name="groq", 
                          model_name="llama-3.3-70b-versatile"):
    """
    Generate a response using RAG-enhanced LLM

    Step 1: Query rewriting into effective search queries using LLM.
    https://arxiv.org/abs/2305.14283

    Returns:
    - answer_text: The generated answer text from LLM provided with context
    - context: The context retrieved from the vector store and used to generate the answer
    """
    

    # Step 1: Query rewriting using the LLM 
    # I use Google Gemini for rewriting queries because it has the biggest free tier limits
    rewrite_prompt = f"""
    You are a search expert. Your task is to rewrite the following medical question into concise and effective search query
    that focus on the key medical concepts from the question.
    The query should be 3-7 words long and target the core medical conditions or treatments.
    
    USER QUESTION: {user_query}
    
    Format your response as one search query without any explanations or numbering.
    """
    rewriter_llm = GoogleGenAI(
            model='gemini-2.0-flash',  
            api_key=google_api_key)
    rewrite_response = rewriter_llm.complete(rewrite_prompt)
    rewritten_queries = rewrite_response.text.strip().split('\n') # If we change prompt to return multiple queries, we can split them by new line

    # Filter out empty queries and strip whitespace
    rewritten_queries = [q.strip() for q in rewritten_queries if q.strip()]

    print(f"Original query: {user_query}")
    print(f"Rewritten queries: {rewritten_queries}")

    # Step 2: Retrieve contexts using all queries
    if provider_name.lower() == "together":
        llm = TogetherLLM(
            model=model_name,
            api_base="https://api.together.xyz/v1",
            api_key=together_api_key,
            is_chat_model=True,
            is_function_calling_model=True,
            temperature=0.1
        )


    elif provider_name.lower() == "groq":
        llm = Groq(model=model_name, 
                   api_key=groq_api_key)
        

    else:
        llm = GoogleGenAI(
            model=model_name,
            api_key=google_api_key  
        )
        


    query_engine = activate_query_engine(similarity_top_k=similarity_top_k, 
                                         sparse_top_k=sparse_top_k,
                                         llm=llm)  
    
    context = query_engine.query(user_query)

    prompt = f"""
        You are a clinically informed mental health decision support system.
            • Provide accurate, evidence-based information relevant to the user’s question.
            • Use the provided context to generate your response.
            • If the available information is insufficient to generate a reliable answer, clearly state that and avoid unsupported assumptions.
            • Ensure your response is precise, concise (no longer than 5-7 sentences), and directly addresses the user’s question.

        CONTEXT: {context}
        USER QUESTION: {user_query}
        """
    
    if provider_name.lower() == "together":
        messages=[ChatMessage(role = "user", content = prompt)]
        # Use non-streaming version to capture the full response
        full_response = llm.chat(messages)
        answer_text = full_response.message.content

    elif provider_name.lower() == "groq":
        full_response = llm.complete(prompt)
        answer_text = str(full_response)

    else:
        full_response = llm.complete(prompt)
        answer_text = full_response.text

    return answer_text, context


def generate_vanilla_response(user_query: str,  
                          provider_name="groq", 
                          model_name="llama-3.3-70b-versatile"):
    """
    Generate a response using vanilla LLM
    Args:
        - user_query: The user's question to be answered by the LLM.
    Returns:
        - answer_text: The generated answer text from LLM 
    
    """
    user_query = user_query

    prompt = f"""
        You are a clinically informed mental health decision support system.
            • Provide accurate, evidence-based information relevant to the user’s question.
            • If the available information is insufficient to generate a reliable answer, clearly state that and avoid unsupported assumptions.
            • Ensure your response is precise, concise (no longer than 5-7 sentences), and directly addresses the user’s question.

        USER QUESTION: {user_query}
        """
    if provider_name.lower() == "together":
        llm = TogetherLLM(
            model=model_name,
            api_base="https://api.together.xyz/v1",
            api_key=together_api_key,
            is_chat_model=True,
            is_function_calling_model=True,
            temperature=0.1
        )
        messages = [ChatMessage(role="user", content=prompt)]
        full_response = llm.chat(messages)
        answer_text = full_response.message.content

    elif provider_name.lower() == "groq":
        llm = Groq(model=model_name, 
                   api_key=groq_api_key)
        full_response = llm.complete(prompt)
        answer_text = str(full_response)

    else:
        llm = GoogleGenAI(
            model=model_name,
            api_key=google_api_key)
        full_response = llm.complete(prompt)
        answer_text = full_response.text

    return answer_text


def process_questions_from_file(file_path, 
                               provider_name,
                               model_name,
                               similarity_top_k=2,
                               sparse_top_k=10,
                               batch_size: int = 1, 
                               max_rows: int = 1000,  
                               timeout_interval = 1,
                               timeout_seconds = 10):
    
    FILE_PATH = file_path
    BATCH_SIZE = batch_size
    PROVIDER = provider_name
    MODEL = model_name
    similarity_top_k = similarity_top_k
    sparse_top_k = sparse_top_k

    # Clean model name to use in filename by replacing slashes with underscores
    # Because Together AI models contain slach in their names and it results in an error
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
        df['Generated RAG Answer'] = None
        df['Retrieved Context'] = None
        df['Top k Similarity'] = None
        df['Top k Sparse'] = None

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
                                                similarity_top_k=similarity_top_k,
                                                sparse_top_k=sparse_top_k,
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
        df.loc[idx, 'Top k Similarity'] = similarity_top_k
        df.loc[idx, 'Top k Sparse'] = sparse_top_k
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

# Example usage

QUESTIONS_FILE = r"C:\\Users\\kuzne\\Documents\\Python_repo\\2025_01_dissertation\\2025_dissertation\\data\\2025-06 hybrid search\\psychiatry_test_dataset.csv"


process_questions_from_file(file_path=QUESTIONS_FILE, 
                               provider_name="groq",
                               model_name="meta-llama/llama-4-maverick-17b-128e-instruct",
                               batch_size=4,
                               max_rows=320,
                               timeout_seconds=20,
                               similarity_top_k=3,
                               sparse_top_k=10)

'''
# TOGETHER AI meta-llama/Llama-3.2-3B-Instruct-Turbo
process_questions_from_file(file_path=QUESTIONS_FILE, 
                               provider_name="together",
                               model_name="meta-llama/Llama-3.2-3B-Instruct-Turbo",
                               batch_size=4,
                               max_rows=400,
                               timeout_seconds=0,
                               similarity_top_k=3,
                               sparse_top_k=10)
'''