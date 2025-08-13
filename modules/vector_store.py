from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
import streamlit as st
import json
import time
from typing import List, Dict, Any
import config

class VectorStore:
    def __init__(self):
        """Initialize Pinecone vector store and OpenAI client"""
        self.openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.pc = Pinecone(api_key=config.PINECONE_API_KEY)
        self.index = None
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize or connect to Pinecone index"""
        try:
            # Check if index exists
            existing_indexes = self.pc.list_indexes()
            index_names = [idx.name for idx in existing_indexes.indexes]
            
            if config.PINECONE_INDEX_NAME not in index_names:
                # Create index if it doesn't exist
                st.info(f"Creating new Pinecone index: {config.PINECONE_INDEX_NAME}")
                self.pc.create_index(
                    name=config.PINECONE_INDEX_NAME,
                    dimension=config.EMBEDDING_DIMENSION,
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                )
                # Wait for index to be ready
                st.info("Waiting for index to be ready...")
                time.sleep(15)
            
            # Connect to index
            self.index = self.pc.Index(config.PINECONE_INDEX_NAME)
            st.success(f"Connected to Pinecone index: {config.PINECONE_INDEX_NAME}")
            
            # Initialize with sample knowledge base if empty
            try:
                stats = self.index.describe_index_stats()
                if stats.total_vector_count == 0:
                    st.info("Loading initial knowledge base...")
                    self._load_initial_knowledge_base()
            except Exception as stats_error:
                st.warning(f"Could not check index stats: {str(stats_error)}")
                
        except Exception as e:
            st.error(f"Failed to initialize Pinecone index: {str(e)}")
            st.error("Please check your Pinecone API key and try again.")
            # Don't raise the exception, allow the app to continue with limited functionality
            self.index = None
    
    def _load_initial_knowledge_base(self):
        """Load initial knowledge base into vector store"""
        try:
            from modules.knowledge_processor import KnowledgeProcessor
            processor = KnowledgeProcessor()
            processor.load_knowledge_base()
        except Exception as e:
            st.warning(f"Failed to load initial knowledge base: {str(e)}")
    
    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for given text using OpenAI"""
        try:
            response = self.openai_client.embeddings.create(
                model=config.EMBEDDING_MODEL,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            st.error(f"Failed to generate embedding: {str(e)}")
            raise e
    
    def upsert_documents(self, documents: List[Dict[str, Any]]):
        """Upsert documents into vector store"""
        try:
            if self.index is None:
                st.warning("Vector store not initialized. Cannot store documents.")
                return False
                
            vectors = []
            for doc in documents:
                # Generate embedding for document content
                embedding = self.get_embedding(doc['content'])
                
                vector = {
                    'id': doc['id'],
                    'values': embedding,
                    'metadata': {
                        'title': doc.get('title', ''),
                        'content': doc['content'],
                        'category': doc.get('category', 'general'),
                        'tags': doc.get('tags', []),
                        'created_at': doc.get('created_at', time.time())
                    }
                }
                vectors.append(vector)
            
            # Upsert in batches
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)
            
            return True
            
        except Exception as e:
            st.error(f"Failed to upsert documents: {str(e)}")
            return False
    
    def similarity_search(self, query: str, k: int = 5, filter_dict: Dict = None) -> List[Dict]:
        """Perform similarity search and return relevant documents"""
        try:
            if self.index is None:
                st.warning("Vector store not initialized. Using fallback search.")
                return self._fallback_search(query)
                
            # Generate query embedding
            query_embedding = self.get_embedding(query)
            
            # Search similar vectors
            search_results = self.index.query(
                vector=query_embedding,
                top_k=k,
                include_metadata=True,
                filter=filter_dict
            )
            
            # Format results
            results = []
            for match in search_results.matches:
                result = {
                    'id': match.id,
                    'score': match.score,
                    'title': match.metadata.get('title', ''),
                    'content': match.metadata.get('content', ''),
                    'category': match.metadata.get('category', 'general'),
                    'tags': match.metadata.get('tags', [])
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            st.error(f"Failed to perform similarity search: {str(e)}")
            return self._fallback_search(query)
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete document from vector store"""
        try:
            self.index.delete(ids=[doc_id])
            return True
        except Exception as e:
            st.error(f"Failed to delete document: {str(e)}")
            return False
    
    def get_index_stats(self) -> Dict:
        """Get vector store statistics"""
        try:
            stats = self.index.describe_index_stats()
            return {
                'total_vectors': stats.total_vector_count,
                'dimension': stats.dimension,
                'index_fullness': stats.index_fullness
            }
        except Exception as e:
            st.error(f"Failed to get index stats: {str(e)}")
            return {}
    
    def _fallback_search(self, query: str) -> List[Dict]:
        """Fallback search when vector store is not available"""
        # Simple keyword-based search in sample knowledge base
        fallback_docs = [
            {
                'id': 'fallback_001',
                'score': 0.7,
                'title': 'General Help',
                'content': 'I understand you need assistance. Let me help you with your request. Please provide more details about your specific issue so I can assist you better.',
                'category': 'general',
                'tags': ['help', 'support']
            },
            {
                'id': 'fallback_002', 
                'score': 0.6,
                'title': 'Account Support',
                'content': 'For account-related issues, please check your email for any notifications and ensure your account information is up to date. Contact support if you need further assistance.',
                'category': 'account',
                'tags': ['account', 'support']
            }
        ]
        return fallback_docs[:2]

    def is_healthy(self) -> bool:
        """Check if vector store is healthy and accessible"""
        try:
            if self.index is None:
                return False
            stats = self.index.describe_index_stats()
            return True
        except:
            return False
