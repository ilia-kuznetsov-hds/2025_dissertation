import os
import gradio as gr
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.llms.together import TogetherLLM
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.llms.groq import Groq
from llama_index.core.llms import ChatMessage
import pandas as pd



together_api_key = os.getenv("TOGETHER_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

DATABASE_PATH = r"C:\\Users\\kuzne\\Documents\\Python_repo\\2025_01_dissertation\\2025_dissertation\\chromadb"

QUESTIONS_FILE_PATH = r"C:\\Users\\kuzne\\Documents\\Python_repo\\2025_01_dissertation\\2025_dissertation\\data\\reviewed_questions\\ptsd_reviewed_questions test.xlsx"
OUTPUT_FILE_PATH = r"C:\\Users\\kuzne\\Documents\\Python_repo\\2025_01_dissertation\\2025_dissertation\\data\\reviewed_questions\\ptsd_reviewed_questions test_with_answers.xlsx"

'''
Part 1. Connect to external vector store (with existing embeddings)
https://docs.llamaindex.ai/en/stable/module_guides/indexing/vector_store_guide/

This is the way to access previously calculated embeddings stored in the index

'''

# https://docs.llamaindex.ai/en/stable/examples/vector_stores/ChromaIndexDemo/
# Basic example including saing to the disc
client = chromadb.PersistentClient(path=DATABASE_PATH)
collection = client.get_or_create_collection(name="articles")
vector_store = ChromaVectorStore(chroma_collection=collection)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)


df = pd.read_excel(QUESTIONS_FILE_PATH)
user_query = 'How do we treat bipolar disorder in adults?'


def generate_rag_response(user_query, provider_name="deepseek"):
    """
    Generate a response using RAG-enhanced LLM
    
    """
    global index
    user_query = user_query
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

        Your response should be comprehensive, accurate, and based on the context provided."""
    
    if provider_name.lower() == "deepseek":
        llm = TogetherLLM(
            model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
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
        llm = Groq(model="llama-3.2-1b-preview", api_key=groq_api_key)
        full_response = llm.complete(prompt)
        answer_text = full_response

    else:
        llm = GoogleGenAI(
            model="gemini-2.0-flash",
            api_key=google_api_key  
        )
        full_response = llm.complete(prompt)
        answer_text = full_response.text

    return answer_text  


def generate_vanilla_response(user_query, provider_name="deepseek"):
    
    user_query = user_query
    prompt = f"""You are an expert assistant. Answer the user question based on your knowledge.
        If you don't have enough information, state that you don't have enough information.
          
          USER QUESTION: {user_query}

        Your response should be comprehensive and accurate."""

    if provider_name.lower() == "deepseek":
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

    elif provider_name.lower() == "groq":
        llm = Groq(model="llama-3.2-1b-preview", api_key=groq_api_key)
        full_response = llm.complete(prompt)
        answer_text = full_response

    else:
        llm = GoogleGenAI(
            model="gemini-2.0-flash",
            api_key=google_api_key  
        )
        full_response = llm.complete(prompt)
        answer_text = full_response.text
    
    return answer_text


def compare_responses(question, provider_dropdown="deepseek"):
    # Prevent to send empty query to LLM
    if not question.strip():
        return "Please enter a question.", "Please enter a question."

    rag_response = generate_rag_response(question, provider_name=provider_dropdown)
    vanilla_response = generate_vanilla_response(question, provider_name=provider_dropdown)
    return rag_response, vanilla_response

#generate_response_from_llm(user_query, index) 


def process_questions_from_csv(file_path, row_index, vector_store, output_file=None):
    df = pd.read_excel(file_path)

    if 'Generated Answer' not in df.columns:
        df['Generated Answer'] = None
    
    # Check if row index is valid
    if row_index < 0 or row_index >= len(df):
        raise ValueError(f"Row index {row_index} is out of bounds (file has {len(df)} rows)")
    
    # Get the question
    question = df.at[row_index, 'Modified Questions']
    print(f"Processing question at row {row_index}: {question}")
 
    answer = generate_rag_response(question, vector_store)

    # Debug check to confirm we have the answer
    print(f"Debug - Captured answer: {answer[:50]}...")

    df.at[row_index, 'Generated Answer'] = answer

    # If no output_file specified, create one based on the input filename
    if output_file is None:
        base_name = os.path.basename(file_path)
        name_without_ext = os.path.splitext(base_name)[0]
        output_file = os.path.join(os.path.dirname(file_path), f"{name_without_ext}_with_answers.xlsx")

    # Save the updated DataFrame
    df.to_excel(output_file, index=False)
    print(f"Results saved to {output_file}")

    return df






'''
process_questions_from_csv(file_path=QUESTIONS_FILE_PATH, 
                           row_index=6, 
                           vector_store=index, 
                           output_file=None)
'''

def create_gradio_interface():
    """Create and launch the Gradio interface.
    
    """

    # Creates the container for organizing UI elements
    # https://www.gradio.app/guides/blocks-and-event-listeners
    with gr.Blocks(theme=gr.themes.Soft(),
                   fill_width=True) as demo:
        gr.Markdown("# LLM Response Comparison: RAG vs. Vanilla")
        
        with gr.Row():
            question_input = gr.Textbox(
                label="Enter your question",
                placeholder="Type your question here...",
                lines=3
            )

        with gr.Row():
            provider_dropdown = gr.Dropdown(
                choices=["deepseek", "groq", "gemini"], 
                label="Select RAG Model", 
                value="deepseek"
            )
        with gr.Row():
            submit_btn = gr.Button("Submit", variant="primary")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### RAG-Enhanced Response")
                rag_response = gr.Textbox(lines=40, label="RAG-Enhanced Response")
            
            with gr.Column():
                gr.Markdown("### Vanilla LLM Response")
                vanilla_response = gr.Textbox(lines=40, label="Vanilla LLM Response")

    
        # Connect the components
        submit_btn.click(
            fn=compare_responses,
            inputs=[question_input, provider_dropdown],
            outputs=[rag_response, vanilla_response]
            )
    
    return demo



demo = create_gradio_interface()
demo.launch(share=False)








