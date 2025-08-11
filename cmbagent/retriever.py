import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from PyPDF2 import PdfReader

class VectorRetriever:
    def __init__(self, model_name='all-MiniLM-L6-v2', chunk_size=512):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []
        self.chunk_size = chunk_size
   
    def chunk_text(self, text):
        """Basic text chunking"""
        return [text[i:i+self.chunk_size] 
               for i in range(0, len(text), self.chunk_size)]
   
    def add_documents(self, documents):
        """Add documents with proper chunking"""
        if not documents:
            raise ValueError("No documents provided")
            
        for doc in documents:
            chunks = self.chunk_text(doc)
            self.documents.extend(chunks)
            
        embeddings = self.model.encode(self.documents)
        
        if self.index is None:
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        
    def search(self, query, top_k=3):
        """Search with error handling"""
        if not self.index:
            raise RuntimeError("Index not initialized - call add_documents() first")
            
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(query_embedding, top_k)
        return [self.documents[i] for i in indices[0]]
    
    @staticmethod
    def pdf_to_text(filepath):
        """Extract text from PDF"""
        with open(filepath, 'rb') as f:
            return '\n'.join(
                page.extract_text() 
                for page in PdfReader(f).pages
            )