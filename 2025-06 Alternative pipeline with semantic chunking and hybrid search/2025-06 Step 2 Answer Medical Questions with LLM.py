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
import json
from llama_index.core import get_response_synthesizer

# Configure global embedding model
# You need to do it, because by default LlmaIndex uses OpenAI embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name='sentence-transformers/all-mpnet-base-v2')

# By default, LlamaIndex's response senthesizer is set to OpenAI, but we want to use Google GenAI as a fallback
google_api_key = os.getenv("GOOGLE_API_KEY")
Settings.llm = GoogleGenAI(model="gemini-2.0-flash")

groq_api_key = os.getenv("GROQ_API_KEY")
together_api_key = os.getenv("TOGETHER_API_KEY")


def activate_query_engine(similarity_top_k=2, sparse_top_k=12, llm = None, response_mode="compact"):
    """
    Activate Qdrant for hybrid search.
    This function initializes the Qdrant client populated with psychiatry guidelines
    and sets up the query engine for hybrid search.
    Args:
        similarity_top_k (int): Number of top similar vectors to return.
        sparse_top_k (int): Number of top sparse vectors to return.
        llm: Language model to use for responses. If None, uses Settings.llm.
        response_mode (str): Response synthesis mode. Options: "compact", "refine", "tree_summarize"
    Returns:
        query_engine (QueryEngine): A query engine configured for hybrid search.

    """
    # Initialize Qdrant client
    client = QdrantClient(url="http://localhost:6333")
    
    # Define collection name
    # The collection name for the last pipeline - "psychiatry_guidelines"
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
    
    # Use provided LLM or fall back to Settings.llm
    response_llm = llm if llm is not None else Settings.llm

    # https://docs.llamaindex.ai/en/stable/module_guides/querying/response_synthesizers/
    # Possible modes: 
    # compact - intermediate option, 
    # refine - uses the most LLM calls, 
    # tree_summarize - tries to stuck everything in one message, the most concise
    response_synthesizer = get_response_synthesizer(
        response_mode=response_mode
    )


    # Create query engine with hybrid search capabilities
    query_engine = index.as_query_engine(
        similarity_top_k=similarity_top_k, 
        sparse_top_k=sparse_top_k, 
        vector_store_query_mode="hybrid",
        llm=response_llm,
        response_synthesizer=response_synthesizer  
    )
    
    return query_engine


def generate_rag_response(user_query: str,  
                          similarity_top_k=5, 
                          sparse_top_k=10,
                          provider_name="groq", 
                          model_name="llama-3.3-70b-versatile",
                          max_output_tokens=8192,
                          response_synthesizer_mode="compact"):
    """
    Generate a response using RAG-enhanced LLM

    Step 1: Query rewriting into effective search queries using LLM.
    https://arxiv.org/abs/2305.14283

    Returns:
    - answer_text: The generated answer text from LLM provided with context
    - context: The context retrieved from the vector store and used to generate the answer
    """
    
    # Step 1: Query rewriting using the LLM 
    rewrite_prompt = f"""
    You are a psychiatric search expert specializing in information retrieval. Your task is to generate search queries 
    to effectively retrieve relevant psychiatric literature based on the user's question.
    
    Follow these guidelines:
    1. **Rank by Importance**: The queries must be ranked by importance with respect to the question, with the first one being the most important.
    2. **Relevance to Question**: Each query may ask for information regarding the answer options but must always be relevant to the question.
    3. **Differentiable and Efficient**: The queries must be differentiable and efficient, ensuring that the aggregate retrieved information from all queries
    provides as much as possible the needed information to arrive at the correct answer.

    USER QUESTION: {user_query}

    Example:
        Question: "A 55-year-old man with a history of myocardial infarction 3 months ago presents with feelings of depression. 
        He says that he has become detached from his friends and family and has daily feelings of hopelessness. 
        He says he has started to avoid strenuous activities and is no longer going to his favorite bar where he used to spend a lot of time 
        drinking with his buddies. The patient says these symptoms have been ongoing for the past 6 weeks, and his wife is starting to worry about his behavior. 
        He notes that he continues to have nightmares that he is having another heart attack. He says he is even more jumpy than he used to be, and 
        he startles very easily. What according to you is the probable diagnosis?"
        Output:
        {{
        "number_of_searches": 3,
        "search_queries": {{
        "1": "PTSD diagnostic criteria",
        "2": "trauma-related disorders symptoms diagnosis",
        "3": "PTSD treatment guidelines therapy"
        }},
        "search_query_goals": {{
        "1": "What are the DSM-5 criteria for PTSD?",
        "2": "How are trauma-related disorders diagnosed?",
        "3": "What are the treatment guidelines for PTSD?"
        }}
        }}

        The output must follow the JSON format as in the above example.
    """

    # Step 2: Retrieve contexts using all queries
    if provider_name.lower() == "together":
        llm = TogetherLLM(
            model=model_name,
            api_base="https://api.together.xyz/v1",
            api_key=together_api_key,
            is_chat_model=True,
            is_function_calling_model=True,
            temperature=0.0, # Set temperature to 0 for deterministic outputs
            max_output_tokens=max_output_tokens  # Maximum output tokens for Together models 
        )

        messages=[ChatMessage(role = "user", content = rewrite_prompt)]
        # Use non-streaming version to capture the full response
        rewrite_query = llm.chat(messages)
        search_queries_json = rewrite_query.message.content.strip()
        

    elif provider_name.lower() == "groq":
        llm = Groq(model=model_name, 
                   api_key=groq_api_key)
        rewrite_query = llm.complete(rewrite_prompt)
        search_queries_json = str(rewrite_query)
        

    else:
        llm = GoogleGenAI(
            model=model_name,
            api_key=google_api_key  
        )
        rewrite_query = llm.complete(rewrite_prompt)
        search_queries_json = rewrite_query.text

    # Parse the JSON response to extract search queries
    try:
        # Try to extract JSON from the response (in case there's extra text)
        start_idx = search_queries_json.find('{')
        end_idx = search_queries_json.rfind('}') + 1
        if start_idx != -1 and end_idx != 0:
            json_str = search_queries_json[start_idx:end_idx]
        else:
            json_str = search_queries_json
            
        queries_data = json.loads(json_str)
        search_queries = list(queries_data["search_queries"].values())
        print(f"Extracted search queries: {search_queries}")
    
    # Fallback to using the original query    
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error parsing JSON response: {e}")
        search_queries = [user_query]
        print(f"Using fallback query: {search_queries}")

    # Step 2: Retrieve contexts using all queries
    query_engine = activate_query_engine(similarity_top_k=similarity_top_k, 
                                         sparse_top_k=sparse_top_k,
                                         llm=llm,
                                         response_mode=response_synthesizer_mode)
    
    long_context = []
    short_contexts = []
    all_sources = []

    for i, query in enumerate(search_queries):
        print(f"Executing search query {i+1}: {query}")
        context = query_engine.query(query)
        # Extract context with source information
        context_with_sources = ""
        sources = set()  # Use set to avoid duplicate sources

        # Get the source nodes from the response
        if hasattr(context, 'source_nodes') and context.source_nodes:
            for j, source_node in enumerate(context.source_nodes):
                node_text = source_node.node.text
                node_metadata = source_node.node.metadata
                
                # Extract source information
                # Unknown - fallback values that appear when the metadata is missing or incomplete
                source_doc = node_metadata.get('source', 'Unknown document')
                page_num = node_metadata.get('page', 'Unknown page')
                source_info = f"[Source: {source_doc}, Page: {page_num}]"
                sources.add(source_info)
                # Add numbered reference for this chunk
                context_with_sources += f"\n[Response {j+1}] {node_text}\n{source_info}\n"
        else:
            # Fallback if no source nodes available
            context_with_sources = str(context)
            sources.add("[Source: Unable to retrieve source information]")
        
        long_context.append(context_with_sources)
        short_contexts.append(str(context))  # Save the result of the query engine's LLM call
        all_sources.extend(list(sources))
    
    # Combine all contexts with clear separation
    combined_context = "\n\n--- Context from different searches ---\n\n".join(short_contexts)
    
    # Create a summary of all sources used
    unique_sources = list(set(all_sources))
    sources_summary = "\n".join([f"• {source}" for source in unique_sources])
     

    prompt = f"""
        You are a clinically informed mental health decision support system and your task is to answer provided question:
            • Carefully analyze provided context and user question.
            • Identify the most relevant information from the context to answer the question. 
            • Think through the problem step-by-step, using only the relevant information to determine the correct answer.
            • If the available information is insufficient to generate a reliable answer, clearly state that and avoid unsupported assumptions.
            • Ensure your response is precise, concise (no longer than 5-7 sentences), and directly addresses the user’s question.

        CONTEXT: {combined_context}
        USER QUESTION: {user_query}
        Please think step-by-step and output your response precise and concise (no longer than 5-7 sentences)

        """
    
    if provider_name.lower() == "together":
        messages=[ChatMessage(role = "user", content = prompt)]
        # Use non-streaming version to capture the full response
        full_response = llm.chat(messages)
        final_answer = full_response.message.content

    elif provider_name.lower() == "groq":
        full_response = llm.complete(prompt)
        final_answer = str(full_response)

    else:
        full_response = llm.complete(prompt)
        final_answer = full_response.text

    return final_answer, combined_context, sources_summary, long_context


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
        You are a clinically informed mental health decision support system and your task is to answer provided question:
            • Carefully analyze provided user question.
            • Identify the most relevant information to answer the question. 
            • Think through the problem step-by-step, using only the relevant information to determine the correct answer.
            • If the available information is insufficient to generate a reliable answer, clearly state that and avoid unsupported assumptions.
            • Ensure your response is precise, concise (no longer than 5-7 sentences), and directly addresses the user’s question.

        USER QUESTION: {user_query}
        Please think step-by-step and output your response precisely and concisely (no longer than 5-7 sentences)
        """
    if provider_name.lower() == "together":
        llm = TogetherLLM(
            model=model_name,
            api_base="https://api.together.xyz/v1",
            api_key=together_api_key,
            is_chat_model=True,
            is_function_calling_model=True,
            temperature=0.0  # Set temperature to 0 for deterministic outputs
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
                               response_synthesizer_mode="compact",
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
    sparse_top_k = sparse_top_k
    
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
        df['Top k Sparse'] = None

        # Add additional fields for JSON format
        if output_format.lower() == "json":
            df['Sources Summary'] = None
            df['Detailed Context'] = None

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
        
        rag_response, context, sources_summary, long_context = generate_rag_response(user_query,
                                                                similarity_top_k=similarity_top_k,
                                                                sparse_top_k=sparse_top_k,
                                                                provider_name=PROVIDER, 
                                                                model_name=MODEL,
                                                                response_synthesizer_mode=response_synthesizer_mode)
        
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
        df.loc[idx, 'Top k Sparse'] = sparse_top_k
        if output_format.lower() == "json":
            df.loc[idx, 'Sources Summary'] = sources_summary
            df.loc[idx, 'Detailed Context'] = str(long_context)  

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



# MAIN EXECUTION


# EVALUATION OF TEST DATASET
# top_k = 5
# sparse_top_k = 10
TRAIN_DATASET_FILE = r"experiments\\test_dataset.json"

# LLAMA 4 SCOUT 

# LLAMA 4 MAVERICK

# GEMMA 3N

'''
process_questions_from_file(file_path=TRAIN_DATASET_FILE,
                            provider_name="together",
                            model_name="meta-llama/Llama-4-Scout-17B-16E-Instruct",
                            similarity_top_k=5,
                            sparse_top_k=10,
                            response_synthesizer_mode="compact",
                            batch_size=10,
                            max_rows=370,
                            timeout_interval=1,
                            timeout_seconds=5,
                            output_format='json') 

'''



process_questions_from_file(file_path=TRAIN_DATASET_FILE,
                            provider_name="together",
                            model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                            similarity_top_k=5,     
                            sparse_top_k=10,
                            response_synthesizer_mode="compact",    
                            batch_size=10,
                            max_rows=370,
                            timeout_interval=1,
                            timeout_seconds=3,
                            output_format='json')

process_questions_from_file(file_path=TRAIN_DATASET_FILE,
                            provider_name="together",
                            model_name="mistralai/Mistral-7B-Instruct-v0.1",
                            similarity_top_k=5,     
                            sparse_top_k=10,
                            response_synthesizer_mode="compact",    
                            batch_size=10,
                            max_rows=370,
                            timeout_interval=1,
                            timeout_seconds=3,
                            output_format='json')
