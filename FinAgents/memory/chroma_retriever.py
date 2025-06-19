from typing import List, Dict, Any, Optional, Union
from sentence_transformers import SentenceTransformer
import json
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


class ChromaRetriever:
    def __init__(self, collection_name: str = "memories", model_name: str = "all-MiniLM-L6-v2"):
        """Initialize ChromaDB retriever.
        
        Args:
            collection_name: Name of the ChromaDB collection.
            model_name: Name of the SentenceTransformer model to use.
        """
        self.client = chromadb.Client(Settings(allow_reset=True)) # allow_reset=True is useful for development/testing.
        self.actual_model_name = model_name  # Store the model name explicitly
        
        # Initialize the embedding function with the stored model name
        self.embedding_function = SentenceTransformerEmbeddingFunction(model_name=self.actual_model_name)
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )
        
    def add_document(self, document: str, metadata: Dict, doc_id: str):
        """Add a document to ChromaDB.
        
        ChromaDB metadata values must be strings, integers, floats, or booleans.
        Lists and nested dictionaries are not directly supported. Therefore,
        list and dict metadata values are serialized to JSON strings. Other types
        are converted to strings.
        
        Args:
            document: Text content to add.
            metadata: Dictionary of metadata.
            doc_id: Unique identifier for the document.
        """
        processed_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, list):
                processed_metadata[key] = json.dumps(value)
            elif isinstance(value, dict):
                processed_metadata[key] = json.dumps(value)
            # Ensure boolean values are handled correctly by ChromaDB if it supports them directly
            # or convert to string if it expects all custom metadata as strings after JSON.
            # For simplicity and to match current logic, converting all non-list/dict to string.
            else:
                processed_metadata[key] = str(value)
                
        self.collection.add(
            documents=[document],
            metadatas=[processed_metadata],
            ids=[doc_id]
        )
        
    def delete_document(self, doc_id: str):
        """Delete a document from ChromaDB.
        
        Args:
            doc_id: ID of document to delete.
        """
        self.collection.delete(ids=[doc_id])
        
    def search(self, query: str, k: int = 5) -> Dict[str, Any]:
        """Search for similar documents.
        
        Args:
            query: Query text.
            k: Number of results to return.
            
        Returns:
            Dict with documents, metadatas, ids, and distances.
            Metadata values that were originally lists or dicts (and stored as JSON strings)
            are deserialized back to their Python types. Numeric strings are converted
            back to numbers.
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=k
        )
        
        # Convert string metadata back to original types where possible
        if 'metadatas' in results and results['metadatas'] and len(results['metadatas']) > 0:
            # Iterate through each query's results (though here we only have one query)
            for i in range(len(results['metadatas'])):
                # Iterate through each document's metadata in the results
                if isinstance(results['metadatas'][i], list):
                    for j in range(len(results['metadatas'][i])):
                        # Process each metadata dictionary
                        if isinstance(results['metadatas'][i][j], dict):
                            metadata_dict = results['metadatas'][i][j]
                            for key, value in metadata_dict.items():
                                if isinstance(value, str):
                                    try:
                                        # Attempt to parse JSON for lists and dicts
                                        if value.startswith('[') and value.endswith(']'):
                                            metadata_dict[key] = json.loads(value)
                                        elif value.startswith('{') and value.endswith('}'):
                                            metadata_dict[key] = json.loads(value)
                                        # Attempt to convert numeric strings back to numbers
                                        # This check is more robust for floats and integers.
                                        elif value.replace('.', '', 1).isdigit():
                                            if '.' in value:
                                                metadata_dict[key] = float(value)
                                            else:
                                                metadata_dict[key] = int(value)
                                        # Add other type conversions if necessary (e.g., 'true'/'false' to boolean)
                                        # For now, if not JSON or number, keep as string.
                                    except (json.JSONDecodeError, ValueError):
                                        # If parsing fails, keep the original string value
                                        pass
                        
        return results