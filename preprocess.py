import pandas as pd
import json
import re
import os
import fitz  # PyMuPDF
from tqdm import tqdm  # For progress bar

def normalize_text(text):
    """Normalize text for better matching"""
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def find_page_for_chunk(doc_path, chunk_text, chunk_index):
    """Find the page number where the chunk appears in the document"""
    try:
        # Open the document
        doc = fitz.open(doc_path)
        
        # Normalize chunk text
        clean_chunk = normalize_text(chunk_text)
        
        # If the chunk text is too long, use just the first 5-10 words
        chunk_words = clean_chunk.split()
        if len(chunk_words) > 10:
            search_text = ' '.join(chunk_words[:10])
        else:
            search_text = clean_chunk
            
        # Try different matching strategies
        for page_num, page in enumerate(doc):
            # Get page text and normalize
            page_text = normalize_text(page.get_text())
            
            # Check if the search text is in the page
            if search_text in page_text:
                return page_num
                
        # If we couldn't find an exact match, try with first 5 words
        if len(chunk_words) > 5:
            search_text = ' '.join(chunk_words[:5])
            for page_num, page in enumerate(doc):
                page_text = normalize_text(page.get_text())
                if search_text in page_text:
                    return page_num
        
        # If all else fails, return the first page
        print(f'Warning: cannot find page for chunk {chunk_index}')
        return 0
    
    except Exception as e:
        print(f"Error processing document: {e}")
        return 0

def preprocess_chunks(chunks_df, doc_mapping):
    """Preprocess all chunks to find their page numbers"""
    # Create a dictionary mapping doc_id to path for faster lookup
    doc_paths = {doc['id']: doc['path'] for doc in doc_mapping}
    
    # Add page number column
    chunks_df['page_num'] = None
    
    # Process each chunk
    print("Processing chunks to find page numbers...")
    for index, row in tqdm(chunks_df.iterrows(), total=len(chunks_df)):
        doc_id = row['doc_id']
        
        # Get document path
        if doc_id in doc_paths:
            doc_path = doc_paths[doc_id]
            
            # Find page number
            if os.path.exists(doc_path):
                page_num = find_page_for_chunk(doc_path, row['chunk'], row['index'])
                chunks_df.at[index, 'page_num'] = page_num
            else:
                print(f"Warning: Document not found: {doc_path}")
        else:
            print(f"Warning: No path found for document ID: {doc_id}")
    
    return chunks_df

def main():
    # Load chunk list
    print("Loading data...")
    chunks_df = pd.read_csv('chunk_list.csv')
    
    # Load document mapping
    with open('document_mapping.json', 'r') as f:
        doc_mapping = json.load(f)
    
    # Preprocess chunks
    enhanced_chunks = preprocess_chunks(chunks_df, doc_mapping)
    
    # Save enhanced data
    print("Saving enhanced chunk data...")
    enhanced_chunks.to_csv('enhanced_chunks.csv', index=False)
    print("Done!")

if __name__ == "__main__":
    main()