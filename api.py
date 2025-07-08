from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from llama_index.vector_stores.opensearch import OpensearchVectorClient, OpensearchVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.vector_stores.types import VectorStoreQueryMode
import torch
from os import getenv
import nest_asyncio
import asyncio
from transformers import AutoTokenizer
import re

# Apply nest_asyncio to avoid runtime errors in Jupyter notebooks
nest_asyncio.apply()

app = FastAPI()

# Define input data model
class QueryRequest(BaseModel):
    query: str

# Initialize HuggingFace embedding model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_model_name = 'BAAI/bge-m3'
embed_model = HuggingFaceEmbedding(model_name=embedding_model_name, max_length=1024, device=device)

# Get embedding dimension
embeddings = embed_model.get_text_embedding("test")
dim = len(embeddings)

# Initialize OpenSearch client and vector store
endpoint = getenv("OPENSEARCH_ENDPOINT", "http://localhost:9200")
idx = getenv("OPENSEARCH_INDEX", "dg_md_index")  # Changed index name to reflect markdown content
client = OpensearchVectorClient(
    endpoint=endpoint,
    index=idx,
    dim=dim,  # Use actual dimension instead of hardcoded 1024
    embedding_field="embedding",
    text_field="content",
    search_pipeline="hybrid-search-pipeline",
)
vector_store = OpensearchVectorStore(client)

# Initialize storage context and vector store index
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex(
    nodes=[], storage_context=storage_context, embed_model=embed_model
)
retriever = index.as_retriever(similarity_top_k=3, vector_store_query_mode=VectorStoreQueryMode.HYBRID)

# Initialize tokenizer for counting tokens
tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

def extract_section_from_metadata(metadata: dict) -> str:
    """Extract section name from metadata based on markdown headers"""
    if "header_text" in metadata:
        return metadata["header_text"]
    
    # Try to find parent header if available
    if "headers" in metadata and metadata["headers"]:
        headers = metadata["headers"]
        if isinstance(headers, list) and len(headers) > 0:
            return headers[-1]  # Get the most specific header
    
    # If no header info is available, try to extract from content
    if "section" in metadata:
        return metadata["section"]
    
    # Fallback to document title or filename if available
    if "title" in metadata:
        return metadata["title"]
    
    # Final fallback
    if "file_path" in metadata:
        filename = metadata["file_path"].split("/")[-1]
        return f"File: {filename}"
    
    return "Unknown Section"

async def retrieve_query(query: str):
    results = retriever.retrieve(query)
    expanded_results = []
    
    for r in results:
        # Get full content of the node
        try:
            expanded_text = r.node.get_content()
        except AttributeError:
            # If get_content() doesn't exist, try text property directly
            expanded_text = getattr(r, "text", str(r))
        
        # Clean up the text by removing excessive newlines or spaces
        expanded_text = re.sub(r'\n{3,}', '\n\n', expanded_text)
        expanded_text = re.sub(r'\s{2,}', ' ', expanded_text)
        
        # Truncate text to fit within token limit if needed
        token_count = count_tokens(expanded_text)
        if token_count > 512:
            # Simple approach: truncate to first 512 tokens
            words = expanded_text.split()
            truncated_text = ""
            current_tokens = 0
            
            for word in words:
                word_tokens = count_tokens(word + " ")
                if current_tokens + word_tokens > 500:  # Leave some margin
                    truncated_text += "..."
                    break
                truncated_text += word + " "
                current_tokens += word_tokens
                
            expanded_text = truncated_text
        
        expanded_results.append((r, expanded_text))
    
    return expanded_results

@app.post("/search")
async def search(request: QueryRequest):
    try:
        results = await asyncio.create_task(retrieve_query(request.query))
        response = []
        total_tokens = 0
        
        for r, expanded_text in results:
            tokens = count_tokens(expanded_text)
            total_tokens += tokens
            
            # Extract section information from metadata
            section = extract_section_from_metadata(r.metadata)
            
            response.append({
                "metadata": r.metadata,
                "text": expanded_text,
                "file_path": r.metadata.get("file_path", "N/A"),
                "tokens": tokens,
                "page_label": section  # Use section instead of page_label for markdown
            })
            
        return {"results": response, "total_tokens": total_tokens}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)