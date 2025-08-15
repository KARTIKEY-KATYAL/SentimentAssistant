import os, sys
try:
	from dotenv import load_dotenv  # type: ignore
	load_dotenv()
except Exception:
	pass

def _secret(name: str, default: str | None = None):
	# 1. Env var
	if name in os.environ:
		return os.environ[name]
	# 2. streamlit secrets if available
	if 'streamlit' in sys.modules:  # avoids import cost when unused
		try:  # pragma: no cover
			import streamlit as st  # type: ignore
			if name in st.secrets:
				return st.secrets[name]
		except Exception:
			pass
	return default

# API Keys
OPENAI_API_KEY = _secret("OPENAI_API_KEY")
PINECONE_API_KEY = _secret("PINECONE_API_KEY")

# Pinecone Configuration
PINECONE_ENVIRONMENT = _secret("PINECONE_ENVIRONMENT", "us-east-1")
PINECONE_INDEX_NAME = _secret("PINECONE_INDEX_NAME", "customer-support-rag")

# Model Configuration
EMBEDDING_MODEL = _secret("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = _secret("CHAT_MODEL", "gpt-4o")  # newest model as of May 13 2024
EMBEDDING_DIMENSION = 1536

# Sentiment Thresholds
ESCALATION_THRESHOLD = float(_secret("ESCALATION_THRESHOLD", "0.3"))
HIGH_PRIORITY_THRESHOLD = float(_secret("HIGH_PRIORITY_THRESHOLD", "0.2"))

# Response Configuration
MAX_CONTEXT_LENGTH = int(_secret("MAX_CONTEXT_LENGTH", "4000"))
MAX_RESPONSE_LENGTH = int(_secret("MAX_RESPONSE_LENGTH", "500"))
TEMPERATURE = float(_secret("TEMPERATURE", "0.7"))

# Startup tuning: skip heavy index init until first query if FAST_INIT set
FAST_INIT = _secret("FAST_INIT", "0") in {"1", "true", "True"}

def warn_if_missing():  # pragma: no cover
	missing = []
	if not OPENAI_API_KEY:
		missing.append("OPENAI_API_KEY")
	if not PINECONE_API_KEY:
		missing.append("PINECONE_API_KEY")
	if missing and 'streamlit' in sys.modules:
		try:
			import streamlit as st  # type: ignore
			st.warning(f"Missing secrets: {', '.join(missing)}. Set them in Streamlit Cloud Secrets.")
		except Exception:
			pass
