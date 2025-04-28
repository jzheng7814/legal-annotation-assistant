import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

def compute_and_save_embeddings():
    print("Loading chunks data...")
    # Load chunk list (either original or enhanced)
    chunks_df = pd.read_csv('enhanced_chunks.csv')
    
    print("Loading sentence transformer model...")
    # Initialize the model
    model = SentenceTransformer('all-mpnet-base-v2')
    
    print("Computing embeddings for all chunks...")
    # Compute embeddings for all chunks
    chunk_texts = chunks_df['chunk'].tolist()
    chunk_embeddings = model.encode(chunk_texts)
    
    print("Saving embeddings to file...")
    # Save embeddings to a file
    with open('chunk_embeddings.pkl', 'wb') as f:
        pickle.dump(chunk_embeddings, f)
    
    print("Done! Embeddings saved to chunk_embeddings.pkl")

if __name__ == "__main__":
    compute_and_save_embeddings()