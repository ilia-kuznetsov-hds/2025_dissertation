import pandas as pd
import re
from typing import Dict
from dataclasses import dataclass
import json


'''
The dataclass decorator automatically generates 
methods like __init__(), __repr__(), and __eq__() for a class.

'''

@dataclass
class Article:
    pmcid: str
    title: str
    keywords: str
    abstract: str
    body: str
    
# metadata: Dictionary storing metadata about the chunk (e.g., size, start and end preview).    
@dataclass
class Chunk:
    text: str
    metadata: Dict 



# Test file - I downloaded ~500 articles about HPV from PubMed using PubGet
# Columns of the file - pmcid, title, keywords, abstract, body

df = pd.read_csv('C:\\Users\\User\\Documents\\Python_projects\\pubget\\pubget_papilloma\\query_d41d8cd98f00b204e9800998ecf8427e\\subset_allArticles_extractedData\\text.csv')

# explode doesn't work 
# df = df.explode('body') 


'''
Part I: Cleaning the text
Next 2 functions are used to clean the text and split it into paragraphs.
The first function 'create_paragraphs' creates a new column 'paragraphs' which is a list of paragraphs. 
The second function 'create_string' creates a new column 'clean_string' which is a string of text.

'''

def create_paragraphs(df):
    """
    Function to clean text and split into paragraphs

    Return:
      Original DataFrame with a new column 'paragraphs' containing a list of paragraphs
    """

    # Create a new column 'paragraphs' to store the cleaned text
    df['paragraphs'] = None

    for x, row in df['body'].items():
        text = row['body'] if isinstance(row['body'], str) else ''
        text = re.sub(r'\n+', '\n', text)
        paragraphs = [re.sub(r'\s+', ' ', p.strip()) for p in text.split('\n')] 

        # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.at.html
        df.at[x, 'paragraphs'] = paragraphs


    return df


def create_string(df):
    """
    Function to create a new column with a cleaned string of text from 'body" column

    Return:
      a new DataFrame with a new column 'clean_string' containing a string of text
    """
    df['clean_string'] = None

    # Replaced iterrows with items()
    for x, row in df['body'].items():
        text = row if isinstance(row,str) else ''
        # RE to remove multiple new lines
        text = re.sub(r'\n+', '\n', text)
        # RE to remove multiple spaces and strip the text from the beginning and the end of paragraph 
        string = [re.sub(r'\s+', ' ', s.strip()) for s in text.split('\n')]
        df.at[x, 'clean_string'] = '\n'.join(s for s in string if s)

    return df


df = create_string(df)

   

"""
Part II: Extracting the sentences 

"""


def extract_sentences(df):
    """
    Function to extract sentences from the 'clean_string' column and 
    create a new column 'sentences' which is a list of sentences

    Split text into sentences while preserving paragraph breaks.

        
    Returns:
        List of sentences

    """

    df['sentences'] = None
    
    
    # Split text into sentences while preserving paragraph breaks.
    for x, row in df['clean_string'].items():
        text = row if isinstance(row, str) else ''
        
        # Split the string into paragraph
        paragraphs = text.split('\n')
        
        # Empty list to save sentences
        sentences = []
        
        
        for paragraph in paragraphs:
        
            # Split paragraph into sentences using Regex
            # (?<=[.!?]) lookbehind ensures the split happens after a ., !, or ?.
            # \s+ -> matches one or more spaces (actual split point).
            # (?=[A-Z]) lookahead ensures the next character is an uppercase letter (likely the start of a new sentence).
            paragraph_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', paragraph.strip())
        
            # Add paragraph break marker
            # This condition ensures that the block of code inside the if statement only executes 
            # if paragraph_sentences is not empty.
            if paragraph_sentences:
                # https://docs.python.org/3/tutorial/datastructures.html
                # Extend the list by appending all the items from the iterable.
                # Adds the extracted sentences to sentences list.
                sentences.extend(paragraph_sentences)
                # Appends "[PARA_BREAK]" as a paragraph separator.
                sentences.append("[PARA_BREAK]")

        # If the last element of sentences is [PARA_BREAK], remove it (to avoid an extra break at the end).
        if sentences and sentences[-1] == "[PARA_BREAK]":
            sentences.pop()
        
        df.at[x, 'sentences'] = sentences

    return df



df = extract_sentences(df) 



'''
Part III: Chunking the articles

Create a new file chunks with metadata.
'''


def create_chunk(df):
    """
    Create overlapping chunks from each article in the DataFrame,
    storing them in df['chunks'] as a list of Chunk objects.
    """

    chunk_size = 1000

    # Initialize 'chunks' column with empty lists to avoid NaN issues
    df['chunks'] = [[] for _ in range(len(df))]

    for i, sentence_list in df['sentences'].items():
        # Prepare to accumulate all chunks from this row
        chunks = []
        current_chunk = []
        current_size = 0

        # Extract metadata for this article (pmcid, title, etc.)
        # Do I need to include abstract and keywords?
        metadata = df.loc[i, ['pmcid', 'title', 'keywords', 'abstract']].to_dict()

        # Process each sentence in the row
        for sentence in sentence_list:
            # Handle paragraph breaks separately
            if sentence == "[PARA_BREAK]":
                # Ensures that a chunk exists before adding paragraph spacing.
                # Prevents unnecessary newlines at the start if current_chunk is empty.
                if current_chunk:
                    current_chunk.append("\n\n")  # Preserve paragraph spacing
                    current_size += 2
                continue

            # Calculate sentence length and consider adding to the chunk
            sentence_size = len(sentence)

            # If adding this sentence exceeds chunk_size, finalize current chunk first
            if current_size + sentence_size > chunk_size and current_chunk:
                # Finalize chunk
                chunk_text = ' '.join(current_chunk).replace(' \n\n ', '\n\n')
                chunk_metadata = {
                    **metadata,
                    'chunk_index': len(chunks),
                    'size': len(chunk_text),
                    'start_text': chunk_text[:50] + '...',
                    'end_text': '...' + chunk_text[-50:]
                }
                chunks.append(Chunk(text=chunk_text, metadata=chunk_metadata))

                # Reset chunk with overlap (keep last sentence for context)
                current_chunk = [current_chunk[-1]] if current_chunk else []
                current_size = len(current_chunk[-1]) if current_chunk else 0

            # Now add the current sentence to the (possibly reset) chunk
            current_chunk.append(sentence)
            current_size += sentence_size + 1  # +1 for the space we add when joining

        # After processing all sentences, finalize any remaining chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk).replace(' \n\n ', '\n\n')
            chunk_metadata = {
                **metadata,
                'chunk_index': len(chunks),
                'size': len(chunk_text),
                'start_text': chunk_text[:50] + '...',
                'end_text': '...' + chunk_text[-50:]
            }
            chunks.append(Chunk(text=chunk_text, metadata=chunk_metadata))

        # Store chunks in the DataFrame
        df.at[i, 'chunks'] = chunks

    return df



            
create_chunk(df)


df.to_csv("articles_paragraphs.csv", index=False)


'''
Saving alternative version of dataset into json file
'''

def chunk_to_dict(chunk):
    return {
        "text": chunk.text,
        "metadata": chunk.metadata
    }

df['chunks'] = df['chunks'].apply(lambda chunk_list: [chunk_to_dict(c) for c in chunk_list])

df.to_json("articles_chunks.json", orient="records")
