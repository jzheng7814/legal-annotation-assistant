from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class ChunkSuggestionEngine(ABC):
    """Abstract base class for chunk suggestion engines."""
    
    @abstractmethod
    def find_relevant_chunks(self, text: str, chunks_df, mapping, k: int = 20):
        """Find relevant chunks for the given text.
        
        Args:
            text: The query text to find chunks for
            chunks_df: DataFrame containing all chunks
            mapping: Document mapping data
            k: Number of chunks to return
            
        Returns:
            List of dictionaries with chunk information
        """
        pass

class EmbeddingsSimilarityEngine(ChunkSuggestionEngine):
    """Engine that finds relevant chunks using embeddings similarity."""
    
    def __init__(self, model, embeddings):
        """Initialize the engine.
        
        Args:
            model: SentenceTransformer model for encoding queries
            embeddings: Pre-computed embeddings for all chunks
        """
        self.model = model
        self.embeddings = embeddings
    
    def find_relevant_chunks(self, text: str, chunks_df, mapping, k: int = 20):
        """Find relevant chunks using embeddings similarity.
        
        Args:
            text: The query text to find chunks for
            chunks_df: DataFrame containing all chunks
            mapping: Document mapping data
            k: Number of chunks to return
            
        Returns:
            List of dictionaries with chunk information
        """
        q = self.model.encode([text])
        sims = cosine_similarity(q, self.embeddings)[0]
        top_idx = np.argsort(sims)[-k:][::-1]
        out = []
        for i in top_idx:
            row = chunks_df.iloc[i]
            doc_id = int(row["doc_id"])
            doc_name = next((d["name"] for d in mapping if d["id"] == doc_id), "Unknown")
            out.append(
                {
                    "index": int(row["index"]),
                    "chunk": row["chunk"],
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "page_num": int(row["page_num"]),
                    "similarity": float(sims[i]),
                }
            )
        return out