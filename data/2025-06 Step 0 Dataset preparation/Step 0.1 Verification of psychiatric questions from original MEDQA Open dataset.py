from llama_index.llms.google_genai import GoogleGenAI
import os
import pandas as pd
import time 
import json

google_api_key = os.getenv("GOOGLE_API_KEY")
dataset_path = r"C:\\Users\\kuzne\\Documents\\Python_repo\\2025_01_dissertation\\2025_dissertation\\data\\2025-05 08.05.25 dataset for classification\\MedQA_open_dataset_classified.xlsx"



def is_clinical_psychiatry_focused(question, model_name='gemini-2.5-flash'):
    """
    Evaluate if the question is truly focused on clinical psychiatry/mental health
    and if the reasoning can be used to answer the question.
    Returns JSON format with classification and reasoning.
    """
    prompt = f"""
    Act as an experienced clinical psychiatrist and medical educator. 

    Question: {question}

    Evaluate provided question and reasoning based on the following criteria:

    CLINICAL PSYCHIATRY FOCUS: Is this question primarily testing knowledge of clinical psychiatry, mental health disorders, psychiatric treatments, or psychological concepts? 
    - Questions that merely mention mental health terms but primarily test other medical knowledge (like diabetes, cardiology, etc.) should be excluded
    - Questions should focus on psychiatric diagnosis, treatment, symptoms, or mental health concepts as the main learning objective

    Provide your response in the following JSON format:
    {{
        "classification": "INCLUDE" or "EXCLUDE",
        "reasoning": "Brief explanation of why you included or excluded this question"
    }}

    Classification options:
    - "INCLUDE" if the question is primarily focused on clinical psychiatry AND the reasoning is useful
    - "EXCLUDE" if the first criteria fail
    """
    
    
    try:
        llm = GoogleGenAI(
            model=model_name,
            api_key=google_api_key  
        )
        response = llm.complete(prompt)
        result_text = response.text.strip()
        print(f"Response: {result_text}")

        # Clean up the response by removing markdown code blocks if present
        if result_text.startswith("```json"):
            result_text = result_text[7:]  # Remove ```json
        if result_text.startswith("```"):
            result_text = result_text[3:]   # Remove ```
        if result_text.endswith("```"):
            result_text = result_text[:-3]  # Remove trailing ```

        # Strip any remaining whitespace
        result_text = result_text.strip()
        print(f"Cleaned Response: {result_text}")

        # Try to parse JSON response
        try:
            result_json = json.loads(result_text)
            
            # Validate the JSON structure
            if "classification" in result_json and "reasoning" in result_json:
                classification = result_json["classification"].lower()
                if classification in ["include", "exclude"]:
                    return {
                        "classification": classification,
                        "reasoning": result_json["reasoning"],
                        "status": "success"
                    }
                else:
                    return {
                        "classification": "re-do",
                        "reasoning": f"Invalid classification received: {classification}",
                        "status": "error"
                    }
            else:
                    return {
                        "classification": "re-do",
                        "reasoning": "Invalid JSON structure received",
                        "status": "error"
                    }
                
        except json.JSONDecodeError:
            return {
                        "classification": "re-do",
                        "reasoning": "JSON error: Unable to parse response",
                        "status": "error"
                    }
        
    except Exception as e:
        print(f"Error evaluating question: {e}")
        return {
            "classification": "re-do",
            "reasoning": f"API Error: {str(e)}",
            "status": "error"
        }        
        

def process_file_json(file_path, batch_size=10, max_rows=150, timeout_interval=25, timeout_seconds=30):
    """
    Process Excel file with is_clinical_psychiatry_focused function and output to JSON.
    Supports resuming from existing JSON output file.
    """
    FILE_PATH = file_path
    BATCH_SIZE = batch_size
    OUTPUT_PATH = FILE_PATH.replace(".xlsx", "_psychiatry_evaluation.json")

    # Check if the output JSON file already exists 
    if os.path.exists(OUTPUT_PATH):
        print(f"Resuming from existing JSON file: {OUTPUT_PATH}")
        with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
            results_data = json.load(f)
        
        # Convert back to DataFrame for processing
        df = pd.DataFrame(results_data)
    else:
        print(f"Starting new evaluation on: {FILE_PATH}")
        df = pd.read_excel(FILE_PATH)
        # Initialize the evaluation columns
        df['psychiatry_classification'] = None
        df['psychiatry_reasoning'] = None
        df['evaluation_status'] = None

    # Filter to psychiatry questions only (if classification column exists)
    if 'classification' in df.columns:
        df = df[df['classification'] == 'psychiatry']
        df = df.reset_index(drop=True)

    # Get rows that don't have evaluation yet
    rows_to_process = df[df['psychiatry_classification'].isna()].index.tolist()

    # Limit to max_rows
    rows_to_process = rows_to_process[:max_rows]
    total_rows = len(rows_to_process)
        
    print(f"Found {len(df[df['psychiatry_classification'].isna()])} unevaluated questions total")
    print(f"Will process {total_rows} questions in this run (max_rows={max_rows})")
    
    for i, idx in enumerate(rows_to_process):
        # Get question
        question = df.loc[idx, 'Modified Questions']
            
        # Skip if empty
        if pd.isna(question):
            df.loc[idx, 'psychiatry_classification'] = "invalid"
            df.loc[idx, 'psychiatry_reasoning'] = "Empty question"
            df.loc[idx, 'evaluation_status'] = "error"
            continue
            
        # Evaluate the question using is_clinical_psychiatry_focused
        evaluation_result = is_clinical_psychiatry_focused(question, model_name='gemini-2.5-flash')
        
        # Extract results from the returned dictionary
        df.loc[idx, 'psychiatry_classification'] = evaluation_result.get('classification', 'error')
        df.loc[idx, 'psychiatry_reasoning'] = evaluation_result.get('reasoning', 'No reasoning provided')
        df.loc[idx, 'evaluation_status'] = evaluation_result.get('status', 'unknown')

        # Handle API errors
        if evaluation_result.get('status') == 'error':
            print(f"Error encountered at row {idx}. Saving progress and continuing...")
        
        # Update progress
        classification = evaluation_result.get('classification', 'error')
        print(f"Processed {i+1}/{total_rows}: Row {idx} - {classification}")

        # Save after each batch
        if (i + 1) % BATCH_SIZE == 0:
            print(f"Saving progress after batch...")
            # Convert DataFrame to JSON and save
            results_json = df.to_dict('records')
            with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
                json.dump(results_json, f, indent=2, ensure_ascii=False)
        
        # Add timeout after specified interval
        if (i + 1) % timeout_interval == 0 and i+1 < total_rows:
            print(f"Taking a {timeout_seconds} second break to avoid rate limits...")
            time.sleep(timeout_seconds)

    # Final save
    print("Saving final results...")
    results_json = df.to_dict('records')
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
    
    # Report completion status
    remaining = len(df[df['psychiatry_classification'].isna()])
    if remaining > 0:
        print(f"Run complete! {total_rows} questions processed. {remaining} questions remain unevaluated.")
    else:
        print(f"All questions have been evaluated! Total: {len(df)} questions.")
    
    # Print summary statistics
    if 'psychiatry_classification' in df.columns:
        include_count = len(df[df['psychiatry_classification'] == 'include'])
        exclude_count = len(df[df['psychiatry_classification'] == 'exclude'])
        error_count = len(df[df['evaluation_status'] == 'error'])
        
        print(f"\nSummary:")
        print(f"INCLUDE: {include_count}")
        print(f"EXCLUDE: {exclude_count}")
        print(f"ERRORS: {error_count}")
    
    return 

# Usage example:
process_file_json(dataset_path, 
                    batch_size=20, 
                    max_rows=1000, 
                    timeout_interval=20, 
                    timeout_seconds=1)

