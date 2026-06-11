import json
import os
import sys
import time
import types
from pathlib import Path

import pandas as pd
from datasets import Dataset
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

evaluate = None
LlamaIndexEmbeddingsWrapper = None
answer_similarity = None


def patch_missing_langchain_vertexai_import():
    try:
        import langchain_community.chat_models.vertexai  # noqa: F401
    except ModuleNotFoundError:
        from langchain_core.language_models.chat_models import BaseChatModel

        class ChatVertexAI(BaseChatModel):
            @property
            def _llm_type(self):
                return "vertexai"

            def _generate(self, *args, **kwargs):
                raise NotImplementedError("VertexAI chat model is not used in this script.")

        module = types.ModuleType("langchain_community.chat_models.vertexai")
        module.ChatVertexAI = ChatVertexAI
        sys.modules["langchain_community.chat_models.vertexai"] = module


def setup_ragas_dependencies():
    global evaluate
    global LlamaIndexEmbeddingsWrapper
    global answer_similarity

    if evaluate is not None:
        return

    patch_missing_langchain_vertexai_import()

    from ragas import evaluate as ragas_evaluate
    from ragas.embeddings import LlamaIndexEmbeddingsWrapper as RagasEmbeddingsWrapper
    from ragas.metrics import answer_similarity as ragas_answer_similarity

    evaluate = ragas_evaluate
    LlamaIndexEmbeddingsWrapper = RagasEmbeddingsWrapper
    answer_similarity = ragas_answer_similarity


# Repository root
REPO_ROOT = Path(__file__).resolve().parents[2]
# Folder path to the directory with model answer files
EXPERIMENTS_PATH = str(REPO_ROOT / "data")


def setup_ragas_evaluator(model_name="gemini-embedding-2"):
    """
    Initialize the RAGAS evaluator with Google Gemini using LlamaIndex.
    """
    setup_ragas_dependencies()

    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")

    gemini_embeddings = GoogleGenAIEmbedding(
        model_name=model_name,
        api_key=google_api_key
    )

    return LlamaIndexEmbeddingsWrapper(gemini_embeddings)


def calculate_answer_similarity(file_path, 
                                answer_type, 
                                model_name="gemini-embedding-2", 
                                max_rows=20, 
                                batch_size=1, 
                                timeout_seconds=10):
    '''
    Calculate answer similarity for vanilla or RAG answers in the dataset.
    '''
    answer_config = {
        "vanilla": {
            "answer_column": "Generated Vanilla Answer",
            "score_column": "Answer Semantic Similarity for vanilla",
            "model_column": "Evaluation Model Answer Semantic Similarity for vanilla",
            "notes_column": "Evaluation Notes Answer Similarity for vanilla",
            "output_suffix": "_vanilla_answer_similarity_evaluated.json",
            "label": "vanilla",
        },
        "rag": {
            "answer_column": "Generated RAG Answer",
            "score_column": "Answer Semantic Similarity for rag",
            "model_column": "Evaluation Model Answer Semantic Similarity for rag",
            "notes_column": "Evaluation Notes Answer Similarity for rag",
            "output_suffix": "_rag_answer_similarity_evaluated.json",
            "label": "RAG",
        },
    }

    if answer_type not in answer_config:
        raise ValueError(f"Unsupported answer_type: {answer_type}. Use 'vanilla' or 'rag'.")

    config = answer_config[answer_type]
    file_path = str(file_path)
    output_path = file_path.replace(".json", config["output_suffix"])

    if os.path.exists(output_path):
        print(f"Resuming from existing file: {output_path}")
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    else:
        print(f"Starting new evaluation on: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        df = pd.DataFrame(data)
        df[config["score_column"]] = None
        df[config["model_column"]] = model_name
        df[config["notes_column"]] = None

    embedding_wrapper = setup_ragas_evaluator(model_name=model_name)

    answer_rows = df[(df[config["answer_column"]].notna()) & 
                        (df[config["score_column"]].isna())]
        
    rows_to_process = answer_rows[:max_rows]
    total_rows = len(rows_to_process)
    print(f"Found {total_rows} {config['label']} answers to evaluate.")

    for i, idx in enumerate(rows_to_process.index):
        try: 
            question = df.loc[idx, 'Modified Questions']
            answer = df.loc[idx, config["answer_column"]]
            ground_truth = df.loc[idx, 'Reasonings']

            data_samples = {
                    'question': [question],
                    'answer': [answer],
                    'ground_truth': [ground_truth]
            }

            dataset = Dataset.from_dict(data_samples)

            score = evaluate(dataset,
                                metrics=[answer_similarity], 
                                embeddings=embedding_wrapper)
            
            score_df = score.to_pandas()
            similarity_score = score_df['semantic_similarity'].iloc[0]

            df.loc[idx, config["score_column"]] = similarity_score
            df.loc[idx, config["model_column"]] = f"Google Gemini {model_name}"

            print(f"Row {idx} - Answer Similarity Score: {score_df}")

        except Exception as e:
            print(f"Error processing row {idx}: {str(e)}")
            df.loc[idx, config["notes_column"]] = f"Error: {str(e)}"
            continue

        if (i + 1) % batch_size == 0:
            print("Saving progress...")
            data = df.to_dict('records')
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            time.sleep(timeout_seconds)

    print(f"Final save to {output_path}...")    
    data = df.to_dict('records')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Evaluation complete! Results saved to {output_path}")

    remaining = len(df[(df[config["answer_column"]].notna()) & 
                        (df[config["score_column"]].isna())])
    if remaining > 0:
        print(f'''Run complete! {total_rows} questions processed. {remaining} questions remain unevaluated.''')
    else:
        print(f"All questions have been evaluated! Total: {len(df)} questions.")


def main():
    test_file = str(
        Path(EXPERIMENTS_PATH)
        / "Meta Llama 4 Maverick 17B-128E-Instruct-FP8.json"
    )

    calculate_answer_similarity(test_file, 
                                answer_type="vanilla", 
                                model_name="gemini-embedding-2", 
                                max_rows=370, 
                                batch_size=10, 
                                timeout_seconds=0)

    calculate_answer_similarity(test_file, 
                                answer_type="rag", 
                                model_name="gemini-embedding-2", 
                                max_rows=370, 
                                batch_size=10, 
                                timeout_seconds=0)


if __name__ == "__main__":
    main()
