import pandas as pd
import textwrap
import hashlib


# Path to the questions dataset
FILE_PATH = 'C:\\Users\\kuzne\\Documents\\Python_repo\\2025_01_dissertation\\2025_dissertation\\data\\MedQA_no-opt_dataset_with_reasoning.xlsx'  # Replace with your file path


# Load the Excel file
df = pd.read_excel(FILE_PATH)


def create_question_id(question, answer):
    """
    Create a unique ID based on complete question and answer content.
    Returns full MD5 hash to minimize collision risk.
    Handles NaN values by converting them to empty strings.
    """
    question = str(question) if not pd.isna(question) else ""
    answer = str(answer) if not pd.isna(answer) else ""
    combined = (question + answer).encode('utf-8')
    return hashlib.md5(combined).hexdigest()  # Return full 32-character hash


def filter_question(df, condition, output_file='filtered_questions_with_ids.xlsx'):
    '''
    Filter questions containing specific condition and save them with unique IDs.
    Skip rows with NaN values in Modified Questions column.
    '''
    mask = df['Reasonings'].str.contains(condition, case=False, na=False)
    # Boolean mask to filter rows with the condition
    condition_df = df[mask].copy()

    if len(condition_df) == 0:
        print(f"No questions found containing '{condition}'")
        return None
    
    # Remove rows with NaN in Modified Questions
    condition_df = condition_df.dropna(subset=['Modified Questions'])

    # Create unique IDs for each question
    condition_df['question_id'] = condition_df.apply(
        lambda row: create_question_id(row['Modified Questions'], row['Reasonings']), 
        axis=1
    )

    # Save to Excel with condition name in filename
    output_filename = f"{condition}_{output_file}"
    condition_df.to_excel(output_filename, index=False)
    
    print(f"Found {len(condition_df)} questions about {condition}")
    print(f"Saved to {output_filename}")
    
    return condition_df


data = filter_question(df, 'anxiety disorder')


def review_condition_questions(condition, 
                               input_file='filtered_questions_with_ids.xlsx', 
                               output_file='reviewed_questions.xlsx',  width=80):
    
    # Try to load previous progress
    # If we already reviewed some questions, we want to keep track of them.
    # This is why we load OUTPUT file first, to check whether it exists.
    excel_path = f'{condition}_{output_file}'
    try:
        previous_df = pd.read_excel(excel_path)
        reviewed_ids = set(previous_df['question_id'])
        print(f"Found {len(reviewed_ids)} previously reviewed questions")
        
        
        # If we don't have any previous results, create an empty DataFrame.
    except FileNotFoundError:
        reviewed_ids = set()
        previous_df = pd.DataFrame()  # Create empty DataFrame for new results
        print("Starting new review session")

    # Load filtered file with questions about the condition.
    # File is created by the filter_question function.
    input_filename = f"{condition}_{input_file}"
    condition_df = pd.read_excel(input_filename)
    
    # Filter out already reviewed questions
    remaining_df = condition_df[~condition_df['question_id'].isin(reviewed_ids)]
    remaining_df['is_appropriate'] = None  # Add a column to mark valid questions

    print(f"Found {len(remaining_df)} questions about {condition}")
    print("For each question, enter:")
    print("1 - Appropriate")
    print("0 - Not appropriate")
    print("q - Quit review")

    for idx, row in remaining_df.iterrows():
        # Wrap question and answer text
        # https://docs.python.org/3/library/textwrap.html
        # Wraps the single paragraph in text, and returns a single 
        # string containing the wrapped paragraph
        question = textwrap.fill(row['Modified Questions'], width=width, 
                               initial_indent="Question: ", subsequent_indent="          ")
        answer = textwrap.fill(row['Reasonings'], width=width,
                             initial_indent="Answer: ", subsequent_indent="        ")
        
        print(f"\n{question}\n")
        print(f"{answer}\n")

        # Part for manual evaluation of the question quality. 
        while True:
            response = input("Is this question appropriate? (1/0/q): ").lower()
            if response in ['1', '0', 'q']:
                break
            print("Invalid input. Please enter 1, 0, or q.")
        
        if response == 'q':
            break
        
        remaining_df.at[idx, 'is_appropriate'] = int(response)


    # Save reviewed questions
    reviewed_df = remaining_df[remaining_df['is_appropriate'].notna()]
    # Separate Dataframe with only appropriate questions
    appropriate_df = reviewed_df[reviewed_df['is_appropriate'] == 1]
    
    if not reviewed_df.empty:
        if not previous_df.empty:
            combined_df = pd.concat([previous_df, reviewed_df], ignore_index=True)

        else:
            combined_df = reviewed_df

        # Save to Excel (OUTPUT path)
        combined_df.to_excel(excel_path, index=False)

        # Append to text file
        mode = 'w' if previous_df.empty else 'a'
        with open(f'{condition}_review_results.txt', mode, encoding='utf-8') as f:
            f.write(f"\nSession: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*50 + "\n\n")

            for _, row in appropriate_df.iterrows():
                wrapped_question = textwrap.fill(row['Modified Questions'], 
                                               width=width,
                                               initial_indent="Question: ",
                                               subsequent_indent="         ")
                wrapped_answer = textwrap.fill(row['Reasonings'], 
                                             width=width,
                                             initial_indent="Answer: ",
                                             subsequent_indent="       ")
                f.write(f"{wrapped_question}\n\n")
                f.write(f"{wrapped_answer}\n\n")
                f.write("-"*width + "\n\n")
            
            # Write session summary
            f.write("\nSESSION SUMMARY\n")
            f.write("="*50 + "\n")
            f.write(f"Questions reviewed this session: {len(reviewed_df)}\n")
            f.write(f"New appropriate questions: {len(appropriate_df)}\n")
            f.write(f"Total reviewed questions: {len(combined_df)}\n\n")


        print(f"\nSaved {len(reviewed_df)} new questions to {excel_path}")
        print(f"Appropriate questions appended to {condition}_review_results.txt")


        # Example usage:
condition_to_review = "anxiety disorder"  # Change this to the condition you want to review
review_condition_questions(condition_to_review,  width=90) 