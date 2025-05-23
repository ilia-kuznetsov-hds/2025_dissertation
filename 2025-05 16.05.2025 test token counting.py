import os
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.llms.together import TogetherLLM
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.llms.groq import Groq
from llama_index.core.llms import ChatMessage
import pandas as pd

import tiktoken
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core import Settings




# Providers API Keys
together_api_key = os.getenv("TOGETHER_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")



token_counter = TokenCountingHandler(
    tokenizer=tiktoken.encoding_for_model("gemini").encode,
    verbose=True,  # set to true to see usage printed to the console
)

Settings.callback_manager = CallbackManager([token_counter])


def generate_vanilla_response(user_query, 
                              provider_name="deepseek", 
                              model_name="DeepSeek-R1-Distill-Llama-70B-free"):
    
    user_query = user_query
    prompt = f"""You are an expert assistant. Answer the user question based on your knowledge.
        If you don't have enough information, state that you don't have enough information.
          
          USER QUESTION: {user_query}

        Your response should be comprehensive and accurate."""

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
        # chat() is the method that sends meggases to LLM and receives the reponses
        # It takes a list of messages as input and returns the full response
        full_response = llm.chat(messages)
        answer_text = full_response.message.content
        usage = full_response.usage_metadata

    elif provider_name.lower() == "groq":
        llm = Groq(model=model_name, 
                   api_key=groq_api_key)
        full_response = llm.complete(prompt)
        answer_text = full_response
        usage = full_response.usage_metadata

    else:
        llm = GoogleGenAI(
            model=model_name,
            api_key=google_api_key
        )
        full_response = llm.complete(prompt)
        answer_text = full_response.text
        
    
    return answer_text



print(generate_vanilla_response("What is the capital of France?",
                          provider_name="gemini", 
                          model_name="gemini-2.0-flash"))


print(
    "Embedding Tokens: ",
    token_counter.total_embedding_token_count,
    "\n",
    "LLM Prompt Tokens: ",
    token_counter.prompt_llm_token_count,
    "\n",
    "LLM Completion Tokens: ",
    token_counter.completion_llm_token_count,
    "\n",
    "Total LLM Token Count: ",
    token_counter.total_llm_token_count,
)