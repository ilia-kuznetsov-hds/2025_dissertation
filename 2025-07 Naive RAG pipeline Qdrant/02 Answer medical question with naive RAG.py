import os
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.together import TogetherLLM
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.llms.groq import Groq
from llama_index.core.llms import ChatMessage
from qdrant_client import QdrantClient
import pandas as pd
import time
import json

# Configure global embedding model (must match ingestion pipeline)
Settings.embed_model = HuggingFaceEmbedding(
    model_name='sentence-transformers/all-mpnet-base-v2')

# API Keys
together_api_key = os.getenv("TOGETHER_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")


_query_engine_cache = {}

def activate_query_engine(similarity_top_k=3):
    """
    Initialize Qdrant vector store connection for querying

    """

    # Check if we already have a cached query engine for this top_k
    if similarity_top_k in _query_engine_cache:
        print(f"Using cached query engine with similarity_top_k={similarity_top_k}")
        return _query_engine_cache[similarity_top_k]
    
    client = QdrantClient(url="http://localhost:6333")
    collection_name = "psychiatry_guidelines_simple_chunking"

    # Check if the collection exists
    try:
        client.get_collection(collection_name)
    except Exception as e:
        print(f"Collection {collection_name} does not exist: {e}")
        return None
    
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name
    )
    
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store, storage_context=storage_context)
    # Create query engine with hybrid search capabilities
    query_engine = index.as_query_engine(
        similarity_top_k=similarity_top_k, 
        response_mode='no_text'    
    )
    # Cache the query engine
    _query_engine_cache[similarity_top_k] = query_engine
    print(f"Query engine created and cached with similarity_top_k={similarity_top_k}")
    return query_engine


_global_query_engine = None

def get_query_engine(similarity_top_k=3):
    """Get or create a global query engine"""
    global _global_query_engine
    
    if _global_query_engine is None:
        print("Creating global query engine...")
        _global_query_engine = activate_query_engine(similarity_top_k=similarity_top_k)
    
    return _global_query_engine



def generate_rag_response(user_query, 
                               provider_name="together", 
                               model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", 
                               top_k=3):
    """
    Generate RAG response without query rewriting - direct semantic search only
    
    Parameters:
    - user_query: The original user question
    - provider_name: LLM provider name
    - model_name: LLM model name
    - top_k: Number of contexts to retrieve
    
    
    Returns:
    - answer_text: Generated answer
    - raw_contexts: List of raw context texts
    - top_k: Actual top_k used
    """
    
    
    
    # Direct semantic search without query rewriting
    query_engine = get_query_engine(similarity_top_k=top_k)
    if query_engine is None:
        print("Failed to create query engine. Please check your Qdrant setup.")
        return None, None
    response = query_engine.query(user_query)
    
    # Extract raw context texts
    raw_contexts = []
    for source_node in response.source_nodes:
        raw_contexts.append(source_node.node.text)
    
    # Combine contexts
    combined_context = "\n\n".join(raw_contexts)
    
    
    
    # Create prompt with context
    prompt = f"""
    You are a clinically informed mental health decision support system.
    • Provide accurate, evidence-based information relevant to the user's question.
    • Use the provided context to generate your response.
    • If the available information is insufficient to generate a reliable answer, clearly state that and avoid unsupported assumptions.
    • Ensure your response is precise, comprehensive, and aligned with current best practices in mental health care.

    CONTEXT: {combined_context}
    USER QUESTION: {user_query}
    """
    
    # Generate response based on provider
    if provider_name.lower() == "together":
        llm = TogetherLLM(
            model=model_name,
            api_base="https://api.together.xyz/v1",
            api_key=together_api_key,
            is_chat_model=True,
            is_function_calling_model=True,
            temperature=0.0
        )
        messages = [ChatMessage(role="user", content=prompt)]
        full_response = llm.chat(messages)
        answer_text = full_response.message.content
        
    elif provider_name.lower() == "groq":
        llm = Groq(model=model_name, api_key=groq_api_key)
        full_response = llm.complete(prompt)
        answer_text = str(full_response)
        
    else:  # Gemini
        llm = GoogleGenAI(model=model_name, api_key=google_api_key)
        full_response = llm.complete(prompt)
        answer_text = full_response.text
    
    return answer_text, combined_context

def generate_vanilla_response(user_query, 
                             provider_name="together", 
                             model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"):
    """
    Generate vanilla response without RAG
    """
    prompt = f"""
    You are a clinically informed mental health decision support system.
    • Provide accurate, evidence-based information relevant to the user's question.
    • If the available information is insufficient to generate a reliable answer, clearly state that and avoid unsupported assumptions.
    • Ensure your response is precise, comprehensive, and aligned with current best practices in mental health care.

    USER QUESTION: {user_query}
    """
    
    if provider_name.lower() == "together":
        llm = TogetherLLM(
            model=model_name,
            api_base="https://api.together.xyz/v1",
            api_key=together_api_key,
            is_chat_model=True,
            is_function_calling_model=True,
            temperature=0.0
        )
        messages = [ChatMessage(role="user", content=prompt)]
        full_response = llm.chat(messages)
        answer_text = full_response.message.content
        
    elif provider_name.lower() == "groq":
        llm = Groq(model=model_name, api_key=groq_api_key)
        full_response = llm.complete(prompt)
        answer_text = str(full_response)
        
    else:  # Gemini
        llm = GoogleGenAI(model=model_name, api_key=google_api_key)
        full_response = llm.complete(prompt)
        answer_text = full_response.text
    
    return answer_text

def process_questions_from_file(file_path, 
                               provider_name,
                               model_name,
                               similarity_top_k=2,
                               batch_size: int = 1, 
                               max_rows: int = 1000,  
                               timeout_interval = 1,
                               timeout_seconds = 10,
                               output_format='csv'):
    
    FILE_PATH = file_path
    BATCH_SIZE = batch_size
    PROVIDER = provider_name
    MODEL = model_name
    similarity_top_k = similarity_top_k
    
    
    # Determine input file type
    input_extension = os.path.splitext(FILE_PATH)[1].lower()

    # Clean model name to use in filename by replacing slashes with underscores
    # Because Together AI models contain slash in their names and it results in an error
    safe_model_name = model_name.replace('/', '_')
    base_name = os.path.basename(FILE_PATH)
    file_name_without_ext = os.path.splitext(base_name)[0]

    file_extension = ".json" if output_format.lower() == "json" else ".csv"
    OUTPUT_PATH = os.path.join(os.path.dirname(FILE_PATH), 
                              f"{file_name_without_ext}_{provider_name}_{safe_model_name}_top{similarity_top_k}_answered{file_extension}")

    # Check if the output file already exists 
    if os.path.exists(OUTPUT_PATH):
        print(f"Resuming from existing file: {OUTPUT_PATH}")
        if output_format.lower() == "json":
            with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            df = pd.DataFrame(data)  # Create DataFrame for consistent processing
        else:
            df = pd.read_csv(OUTPUT_PATH)
        
    else:
        print(f"Starting new processing on: {FILE_PATH}")
        # Read input file based on extension
        if input_extension == '.json':
            with open(FILE_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        elif input_extension == '.csv':
            df = pd.read_csv(FILE_PATH)
        else:
            raise ValueError(f"Unsupported file format: {input_extension}. Only .csv and .json are supported.")
        
        df['Model'] = MODEL
        df['Provider'] = PROVIDER
        df['Generated Vanilla Answer'] = None
        df['Generated RAG Answer'] = None
        df['Retrieved Context'] = None
        df['Top k Similarity'] = None
        


    def save_data(df, output_path, output_format):
        """
        Save DataFrame to either CSV or JSON format
        
        Args:
            df: pandas DataFrame to save
            output_path: path where to save the file
            output_format: "csv" or "json"
        """
        if output_format.lower() == "json":
            # Convert DataFrame to list of dictionaries for JSON
            data = df.to_dict('records')
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        else:
            df.to_csv(output_path, index=False)

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
            save_data(df, OUTPUT_PATH, output_format)
            return
        
        rag_response, context = generate_rag_response(user_query,
                                                                top_k=similarity_top_k,
                                                                provider_name=PROVIDER, 
                                                                model_name=MODEL)
        
         # Handle API errors
        if rag_response is None:
            print(f"Error encountered. Saving progress and exiting.")
            save_data(df, OUTPUT_PATH, output_format)
            return
        
        # Update the dataframe
        df.loc[idx, 'Generated Vanilla Answer'] = vanilla_response
        df.loc[idx, 'Generated RAG Answer'] = rag_response
        df.loc[idx, 'Retrieved Context'] = context
        df.loc[idx, 'Top k Similarity'] = similarity_top_k
         

        print(f"Processed {i+1}/{total_rows}: Row {idx} - Vanilla Answer: {vanilla_response[:50]}...")

        # Save after each batch
        if (i + 1) % BATCH_SIZE == 0:
            print(f"Saving progress after batch...")
            save_data(df, OUTPUT_PATH, output_format)
        
        # Add timeout after specified interval
        if (i + 1) % timeout_interval == 0 and i+1 < total_rows:
            print(f"Taking a {timeout_seconds} second break to avoid rate limits...")
            time.sleep(timeout_seconds)

    # Final save
    save_data(df, OUTPUT_PATH, output_format)
    print(f"Results saved to {OUTPUT_PATH}")
    return None


file_path = 'experiments/naive_rag/test_dataset.json'







process_questions_from_file(file_path, 
                               provider_name="together",
                                    model_name="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                                    similarity_top_k=3,
                                        batch_size=5,
                                        max_rows=370,
                                        output_format='json',
                                        timeout_interval=1,
                                        timeout_seconds=0
                                        )

process_questions_from_file(file_path,
                               provider_name="together",
                                    model_name="meta-llama/Llama-3.2-3B-Instruct-Turbo",
                                    similarity_top_k=3,
                                        batch_size=5,
                                        max_rows=370,
                                        output_format='json',
                                        timeout_interval=1,
                                        timeout_seconds=0
                                        )

process_questions_from_file(file_path,
                               provider_name="together",
                                    model_name="google/gemma-3n-E4B-it",
                                    similarity_top_k=3,
                                        batch_size=5,
                                        max_rows=370,
                                        output_format='json',
                                        timeout_interval=1,
                                        timeout_seconds=0
                                        )