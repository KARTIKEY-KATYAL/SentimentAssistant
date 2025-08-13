import os
try:
	from dotenv import load_dotenv  # type: ignore
	load_dotenv()
except Exception:
	pass

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "default_openai_key")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "default_pinecone_key")

# Pinecone Configuration
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "customer-support-rag")

# Model Configuration
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o"  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
EMBEDDING_DIMENSION = 1536

# Sentiment Thresholds
ESCALATION_THRESHOLD = 0.3  # Negative sentiment threshold for escalation
HIGH_PRIORITY_THRESHOLD = 0.2  # Very negative sentiment threshold

# Response Configuration
MAX_CONTEXT_LENGTH = 4000
MAX_RESPONSE_LENGTH = 500
TEMPERATURE = 0.7
