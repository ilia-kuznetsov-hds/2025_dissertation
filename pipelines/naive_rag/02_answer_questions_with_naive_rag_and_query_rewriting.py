import importlib.util
import os
import time
from pathlib import Path

import pandas as pd
from llama_index.core import VectorStoreIndex
from llama_index.core.llms import ChatMessage
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.llms.openai import OpenAI

# Providers API Keys
together_api_key = os.getenv("TOGETHER_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

# Repository root
REPO_ROOT = Path(__file__).resolve().parents[2]
# Path to ChromaDB persistent client
DATABASE_PATH = str(REPO_ROOT / "chromadb")
QUESTIONS_FILE_PATH = str(REPO_ROOT / "data" / "test_dataset.csv")
NAIVE_RAG_INGEST_PATH = Path(__file__).with_name("01_naive_rag_ingest_pdfs_to_chroma.py")


# Load helpers from the ingest script, whose filename starts with a number.
def load_initialize_vector_store():
    spec = importlib.util.spec_from_file_location(
        "naive_rag_ingest_pdfs_to_chroma",
        NAIVE_RAG_INGEST_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.configure_embedding_model, module.initialize_vector_store


configure_embedding_model, initialize_vector_store = load_initialize_vector_store()


def initialize_query_index(database_path):
    # Register the same embedding model used during ingestion. This matters because
    # query embeddings must live in the same vector space as the stored document
    # embeddings; otherwise similarity search results will be unreliable.
    configure_embedding_model()

    # Reopen the persisted Chroma vector store created by the ingest step.
    # The returned storage context gives LlamaIndex access to the underlying
    # vector_store object without rebuilding embeddings from the source PDFs.
    storage_context = initialize_vector_store(database_path)

    # Wrap the existing vector store in a LlamaIndex query index. This index is
    # only a query interface here; the vectors already exist in Chroma.
    index = VectorStoreIndex.from_vector_store(
        vector_store=storage_context.vector_store
    )
    return storage_context, index



def generate_rag_response(user_query, 
                          provider_name="gemini", 
                          model_name="gemini-3-flash-preview", 
                          top_k = 3,
                          context_window_size = 8192):
    """
    Generate a response using a RAG-enhanced LLM.

    Workflow:
    1. Rewrite the original question into a concise search query using Gemini.
    This follows the query rewriting idea from:
    https://arxiv.org/abs/2305.14283
    2. Retrieve the most relevant context chunks from the vector store.
    3. Pass the retrieved context and original question to the answer-generation LLM.

    Parameters:
    - user_query: The original user question to be answered.
    - provider_name: Name of the answer-generation LLM provider (e.g., "together", "gemini").
    - model_name: Name of the answer-generation model to use.
    - top_k: Number of top contexts to retrieve from the vector store. If the model has
        a small context window, this is reduced to 1.
    - context_window_size: Context window size of the answer-generation model.

    Returns:
    - answer_text: The generated answer from the LLM.
    - source_texts: The retrieved context chunks used to generate the answer.
    - top_k: The final top_k value used after context-window adjustment.
    """
    global index

    # Determine top_k based on context window size
    # Consider context window < 8192 tokens as small
    top_k = top_k
    is_small_context = False
    if context_window_size < 8192:
        top_k = 1
        is_small_context = True
    

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
            model='gemini-3-flash-preview',
            api_key=google_api_key)
    rewrite_response = rewriter_llm.complete(rewrite_prompt)
    rewritten_queries = rewrite_response.text.strip().split('\n') # If we change prompt to return multiple queries, we can split them by new line

    # Filter out empty queries and strip whitespace
    rewritten_queries = [q.strip() for q in rewritten_queries if q.strip()]

    print(f"Original query: {user_query}")
    print(f"Rewritten queries: {rewritten_queries}")

    # Step 2: Retrieve contexts using all queries
    retriever = index.as_retriever(similarity_top_k=1)  # Reduced top_k since 1) we'll get multiple queries; 2) some models have small context window;
    source_texts = []
    used_source_ids = set()  # To track unique sources and avoid duplication
    
    for query in rewritten_queries:
        source_nodes = retriever.retrieve(query)
        for source_node in source_nodes:
            # Only add unique source texts using node ID as identifier
            node_id = source_node.node.node_id
            if node_id not in used_source_ids:
                used_source_ids.add(node_id)
                source_texts.append(source_node.node.text)
    
    context = "\n\n".join(source_texts)

    # Additional context truncation for very small context windows
    if is_small_context and len(context) > 2000:  # Roughly 500-600 tokens
        context = context[:2000] + "..."
        print(f"Context truncated due to small context window ({context_window_size} tokens)")
    
    prompt = f"""
        You are a clinically informed mental health decision support system.
            • Provide accurate, evidence-based information relevant to the user’s question.
            • Use the provided context to generate your response.
            • If the available information is insufficient to generate a reliable answer, clearly state that and avoid unsupported assumptions.
            • Ensure your response is precise, comprehensive, and aligned with current best practices in mental health care.

        CONTEXT: {context}
        USER QUESTION: {user_query}
        """
    
    if provider_name.lower() == "together":
        llm = OpenAI(
            model=model_name,
            api_base="https://api.together.xyz/v1",
            api_key=together_api_key,
            temperature=0.0
        )

        messages=[ChatMessage(role = "user", content = prompt)]
        # Use non-streaming version to capture the full response
        full_response = llm.chat(messages)
        answer_text = full_response.message.content

    else:
        llm = GoogleGenAI(
            model=model_name,
            api_key=google_api_key  
        )
        full_response = llm.complete(prompt)
        answer_text = full_response.text

    return answer_text, source_texts, top_k


def generate_vanilla_response(user_query, 
                              provider_name="gemini", 
                              model_name="gemini-3-flash-preview"):
    """
    Generate a response using only the LLM, without retrieving external context.

    This is the baseline answer-generation function. It sends the original user
    question directly to the selected model, so the response depends only on the
    model's internal knowledge and the system-style instructions in the prompt.

    Parameters:
    - user_query: The original user question to be answered.
    - provider_name: Name of the LLM provider (e.g., "together", "gemini").
    - model_name: Name of the model to use for answer generation.

    Returns:
    - answer_text: The generated answer from the LLM.
    """
    
    user_query = user_query
    prompt = f"""
        You are a clinically informed mental health decision support system.
            • Provide accurate, evidence-based information relevant to the user’s question.
            • If the available information is insufficient to generate a reliable answer, clearly state that and avoid unsupported assumptions.
            • Ensure your response is precise, comprehensive, and aligned with current best practices in mental health care.

        USER QUESTION: {user_query}
        """

    if provider_name.lower() == "together":
        llm = OpenAI(
            model=model_name,
            api_base="https://api.together.xyz/v1",
            api_key=together_api_key,
            temperature=0.0
        )
        messages = [ChatMessage(role="user", content=prompt)]
        # chat() is the method that sends messages to LLM and receives the reponses
        # It takes a list of messages as input and returns the full response
        full_response = llm.chat(messages)
        answer_text = full_response.message.content

    else:
        llm = GoogleGenAI(
            model=model_name,
            api_key=google_api_key  
        )
        full_response = llm.complete(prompt)
        answer_text = full_response.text
    
    return answer_text


def process_questions_from_csv(file_path, 
                               provider_name= "gemini",
                               model_name= "gemini-3-flash-preview",
                               top_k = 3,
                               context_window_size = 8192,
                               batch_size: int = 1, 
                               max_rows: int = 10,  
                               timeout_interval = 1,
                               timeout_seconds = 10):
    """
    Process questions from a CSV file and generate answers using LLM.
    Parameters:
    - file_path: Path to the CSV file containing questions
    - provider_name: Name of the LLM provider (e.g., "together", "gemini")
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
    TOP_K = top_k
    CONTEXT_WINDOW_SIZE = context_window_size

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
        df['Context Window Size'] = CONTEXT_WINDOW_SIZE
        df['Generated Vanilla Answer'] = None
        df['Generated RAG Answer'] = None
        df['Retrieved Context'] = None

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
            df.loc[idx, 'Status'] = "error"
            # Skip to the next iteration
            print(f"Skipping empty question at row {idx}")
            continue
        
        try:
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
            
            # I cannot log the top k number without assigning it to the variable
            # because it is a way to handle the models with small context window size
            # See the generate_rag_response function for details
            rag_response, context, top_k_number = generate_rag_response(user_query,
                                                    provider_name=PROVIDER, 
                                                    model_name=MODEL,
                                                    top_k=TOP_K,
                                                    context_window_size=CONTEXT_WINDOW_SIZE)
        
            # Handle API errors
            if rag_response is None:
                print(f"Error encountered. Saving progress and exiting.")
                df.to_csv(OUTPUT_PATH, index=False)
                return

            # Update the dataframe
            df.loc[idx, 'Generated Vanilla Answer'] = vanilla_response
            df.loc[idx, 'Generated RAG Answer'] = rag_response
            df.loc[idx, 'Top K'] = top_k_number
            df.loc[idx, 'Retrieved Context'] = context
            df.loc[idx, 'Status'] = "completed"
            print(f"Processed {i+1}/{total_rows}: Row {idx} - Vanilla Answer: {vanilla_response[:50]}...")

        except Exception as e:
            error_message = str(e)
            print(f"Error at row {idx}: {error_message}")
            
            # Check for rate limit errors - these should stop execution
            rate_limit_indicators = [
                "rate_limit_exceeded",
                "rate limit reached", 
                "429",
                "quota exceeded",
                "too many requests"
            ]

            if any(indicator in error_message.lower() for indicator in rate_limit_indicators):
                print("RATE LIMIT DETECTED - Stopping execution to avoid further issues...")

                # Mark current row with rate limit error
                df.loc[idx, 'Generated Vanilla Answer'] = "rate_limit_error"
                df.loc[idx, 'Generated RAG Answer'] = "rate_limit_error"
                df.loc[idx, 'Top K'] = "rate_limit_error"
                df.loc[idx, 'Retrieved Context'] = "rate_limit_error"
                df.loc[idx, 'Status'] = "rate_limit_error"
                
                # Save progress and exit
                df.to_csv(OUTPUT_PATH, index=False)
                print(f"Progress saved to {OUTPUT_PATH}")
                print(f"Processed {i} rows before hitting rate limit")
                print("Wait for rate limit reset, then run again to continue")
                return  # Exit the function

            else:
                # For all other errors, mark the row and continue processing
                df.loc[idx, 'Generated Vanilla Answer'] = "error"
                df.loc[idx, 'Generated RAG Answer'] = "error"
                df.loc[idx, 'Top K'] = "error"
                df.loc[idx, 'Retrieved Context'] = "error"
                df.loc[idx, 'Status'] = "error"
                print(f"Non-critical error at row {idx}: {e}  - continuing with next row")

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
Main script execution.

This section runs only when using the file as a standalone pipeline script.
It initializes the vector-store-backed query index, then processes questions
from the test dataset and writes the generated vanilla and RAG answers to CSV.
'''


# Initialize the vector store
storage_context, index = initialize_query_index(DATABASE_PATH)

# Test run for Gemini model
process_questions_from_csv(QUESTIONS_FILE_PATH,
                            provider_name='gemini',
                            model_name="gemini-3-flash-preview",
                            context_window_size=1048576,
                            batch_size=1,
                            max_rows=3,
                            timeout_seconds=1,
                            )
