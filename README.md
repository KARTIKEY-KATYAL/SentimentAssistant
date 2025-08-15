<div align="center">

# Customer Support RAG with Sentiment, Escalation & Satisfaction Intelligence

Robust Retrieval-Augmented Generation (RAG) platform for customer support combining real‑time sentiment & emotion analysis, escalation risk prediction, empathetic tone control, and satisfaction feedback loops.

<p><strong>Status:</strong> Ready for deployment · <strong>Stack:</strong> Streamlit · OpenAI · Pinecone · RAG Evaluation · Analytics Dashboard</p>

</div>

## 🚀 Live Demo
Add the deployed public URL before submission:
```
https://<your-live-app-url>
```
Status: Pending deployment link.

## 🌐 Domain Focus
**Domain:** SaaS Customer Support / Customer Success.

Knowledge base content models typical SaaS help center topics: authentication & password reset, billing & subscription management, security & privacy, account recovery, and API integration. The system is optimized for:
- High‑volume live chat interactions
- Emotionally charged incidents (billing issues, access lockouts)
- Early escalation / churn risk detection via sentiment trajectory & lexical triggers
- Blending empathetic acknowledgement with procedural guidance

Adaptation guidance for other regulated domains (healthcare, finance, legal):
1. Replace / extend `data/` articles with domain-specific sources & adjust chunk size (e.g., longer legal clauses → smaller chunk_size, higher overlap).
2. Add compliance guardrails (PII/PHI scrubbing; policy filters before LLM calls).
3. Tune escalation keyword lists & thresholds to domain risk vocabulary.
4. Optionally swap OpenAI models for self‑hosted embeddings + LLM where data residency required.


## 📋 Features

### Core RAG Capabilities
- **Knowledge Base Processing**: Intelligent chunking and indexing of help articles
- **Semantic Search**: OpenAI embeddings with Pinecone vector database for accurate document retrieval
- **Context-Aware Responses**: GPT-4o powered responses using retrieved knowledge base context

### Advanced Customer Support Features
- **Real-time Sentiment Analysis**: Detects customer emotions (anger, frustration, satisfaction) with confidence scores
- **Escalation Prediction**: AI-powered risk assessment identifying customers likely to escalate issues
- **Empathetic Response Generation**: Tone calibration based on customer emotional state
- **Multi-turn Conversation Handling**: Maintains context across conversation history
- **Customer Satisfaction Tracking**: In-conversation feedback (1–5 rating) with trend analysis and average satisfaction metric

### Evaluation & Analytics
- **RAGAS Metrics**: Context precision, faithfulness, answer relevancy, and retrieval accuracy
- **Performance Monitoring**: Response latency, sentiment trends, escalation alerts
- **Real-time Dashboard**: Conversation analytics, system health monitoring

## 🛠 Technology Stack

- **Frontend/Deployment**: Streamlit (deployed on Replit)
- **Embedding Models**: OpenAI text-embedding-3-small
- **Vector Database**: Pinecone
- **Language Models**: OpenAI GPT-4o
- **Evaluation**: RAGAS-inspired metrics framework

## 🏗 System Architecture

```
Customer Message → Sentiment Analysis → Knowledge Retrieval → Response Generation
                ↓                      ↓                    ↓
            Escalation Check    →   Context Precision   →   Tone Calibration
                ↓                      ↓                    ↓
            Alert System        →   Quality Evaluation  →   Customer Response
```

### Core Components

1. **Vector Store Module**: Manages Pinecone integration and semantic search (uses word-level chunking with overlap for longer articles)
2. **Sentiment Analyzer**: Real-time emotion and sentiment detection
3. **Escalation Predictor**: Risk assessment and escalation alerts
4. **Response Generator**: Context-aware, empathetic response creation
5. **Knowledge Processor**: Article management and indexing
6. **RAG Evaluator**: Quality metrics and performance assessment

## ⚙️ Quick Start (Local)

### Prerequisites
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))
- Pinecone API key ([Get one here](https://www.pinecone.io/))

### 1. Clone
```bash
git clone https://github.com/your-user/sentiment-assistant.git
cd sentiment-assistant/SentimentAssistant
```

### 2. Environment Variables
Create a `.env` from the provided template:
```bash
copy .env.example .env  # Windows
# cp .env.example .env  # macOS/Linux
```
Fill it with your real keys:
```
OPENAI_API_KEY=...
PINECONE_API_KEY=...
PINECONE_ENVIRONMENT=us-east-1
PINECONE_INDEX_NAME=customer-support-rag
```

### 3. Install
```bash
pip install -r requirements.txt
```

### 4. Run
```bash
streamlit run app.py
```

### Optional: Docker
```bash
docker build -t customer-support-rag .
docker run -p 8501:8501 --env-file .env customer-support-rag
```
App at: http://localhost:8501

### Environment Variables
| Name | Description |
|------|-------------|
| OPENAI_API_KEY | OpenAI API key |
| PINECONE_API_KEY | Pinecone API key |
| PINECONE_ENVIRONMENT | Region (e.g. us-east-1) |
| PINECONE_INDEX_NAME | Index name (auto-created) |
| EMBEDDING_MODEL | Override (optional) |
| CHAT_MODEL | Override (optional) |

## 📊 Usage Examples

### Customer Support Conversation
```
Customer: "I'm really frustrated! My password reset isn't working and I've been trying for hours!"

System Analysis:
- Sentiment: Negative (0.2/1.0) - Frustration detected
- Escalation Risk: High (0.8/1.0) - Immediate attention needed
- Knowledge Retrieved: Password reset troubleshooting guide

AI Response: "I completely understand your frustration, and I sincerely apologize for the trouble you're experiencing with the password reset. Let me help you resolve this immediately..."
```

### Analytics Dashboard Features
- **Sentiment Trends**: Visual charts showing customer mood over time
- **Escalation Alerts**: Real-time notifications for high-risk conversations
- **Performance Metrics**: Response times, retrieval accuracy, system health
- **Satisfaction Metrics**: Average rating and trend (improving / stable / declining)
- **Conversation History**: Detailed logs with sentiment and escalation data

## 🔧 Configuration

### Model Settings
```python
# OpenAI Configuration
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o"
EMBEDDING_DIMENSION = 1536

# Sentiment Thresholds
ESCALATION_THRESHOLD = 0.3
HIGH_PRIORITY_THRESHOLD = 0.2

# Response Configuration
MAX_RESPONSE_LENGTH = 500
TEMPERATURE = 0.7
```

### Chunking Strategy
Articles are split into ~120-word chunks with a 20-word overlap to preserve context across chunk boundaries. Each chunk is stored with parent_id, chunk_index, and total_chunks metadata to enable future reconstruction and hierarchical reasoning.

### Pinecone Setup
```python
# Index Configuration
PINECONE_INDEX_NAME = "customer-support-rag"
PINECONE_ENVIRONMENT = "us-east-1"
```

## 📈 Evaluation Metrics

The system implements comprehensive evaluation using RAGAS-inspired metrics:

### Context Quality
- **Context Precision**: Relevance of retrieved documents (0.0-1.0)
- **Context Recall**: Coverage of necessary information (0.0-1.0)

### Response Quality  
- **Faithfulness**: Accuracy to source material (0.0-1.0)
- **Answer Relevancy**: Relevance to customer query (0.0-1.0)

### System Performance
- **Retrieval Accuracy**: Quality of document matching (0.0-1.0)
- **Response Latency**: Average response generation time

### Example Evaluation Output
```json
{
  "metrics": {
    "context_precision": 0.85,
    "context_recall": 0.78,
    "faithfulness": 0.92,
    "answer_relevancy": 0.88,
    "retrieval_accuracy": 0.81
  },
  "overall_score": 0.85
}
```

## 🎯 Key Technical Challenges Solved

### 1. Emotion Detection in Text
- Multi-dimensional sentiment analysis beyond positive/negative
- Emotion categorization (anger, frustration, confusion, satisfaction)
- Confidence scoring and uncertainty handling

### 2. Context-Aware Empathy Modeling
- Dynamic tone adjustment based on customer emotional state
- Escalation-appropriate response templates
- Cultural sensitivity in language patterns

### 3. Escalation Prediction Algorithms
- Historical conversation pattern analysis
- Real-time risk scoring using multiple indicators
- Proactive escalation alerts and recommendations

### 4. Multi-turn Conversation Analysis
- Conversation state management and persistence
- Sentiment trend analysis across message history
- Context preservation for coherent responses

### 5. Response Tone Calibration
- Adaptive communication style selection
- Empathy level adjustment based on customer needs
- Professional yet personalized response generation

## 📁 Project Structure

```
├── app.py                        # Main Streamlit application
├── config.py                     # Configuration settings
├── modules/                      # Core logic modules
│   ├── vector_store.py           # Pinecone integration
│   ├── sentiment_analysis.py     # Emotion detection
│   ├── escalation_predictor.py   # Risk assessment
│   ├── response_generator.py     # AI response creation
│   ├── knowledge_processor.py    # Knowledge base management
│   ├── evaluation.py             # RAG evaluation metrics
│   └── customer_satisfaction.py  # Satisfaction tracking
├── utils/
│   └── helpers.py               # Utility helpers
├── data/
│   └── sample_knowledge_base.json # Sample help articles
├── tests/
│   ├── test_chunking.py         # Chunking strategy tests
│   └── test_satisfaction.py     # Satisfaction tracker tests
├── evaluate.py                  # Batch evaluation CLI
├── LICENSE                      # MIT license
├── .streamlit/
│   └── config.toml              # Streamlit configuration
├── .env.example                 # Environment template
├── Dockerfile                   # Container build definition
└── requirements.txt             # Python dependencies
```

## 🚀 Deployment Options

### 1. Streamlit Community Cloud
1. Push to public GitHub.
2. Go to https://share.streamlit.io → New app.
3. Add secrets (Settings → Secrets):
```
OPENAI_API_KEY="sk-..."
PINECONE_API_KEY="..."
PINECONE_ENVIRONMENT="us-east-1"
PINECONE_INDEX_NAME="customer-support-rag"
```
4. Deploy (first run creates Pinecone index; may take ~15s).

### 2. Hugging Face Spaces (Docker)
1. Create Space → Type: Docker.
2. Include `Dockerfile`, `requirements.txt`, `app.py`.
3. Add secrets under Variables.
4. Build & launch.

### 3. Replit
1. Import repo.
2. Add secrets via lock icon.
3. Run:
```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

### 4. Docker on VPS
```bash
docker build -t csr-app .
docker run -d --name csr -p 80:8501 --env-file .env csr-app
```

### Production Tips
- Rotate API keys, restrict usage.
- Add auth for public demo (basic auth / reverse proxy).
- Monitor Pinecone usage (vector count & QPS limits).
- Cache embeddings (knowledge base rarely changes).

## 🔐 Security & Privacy
- Avoid logging PII.
- Redact sensitive info before sending to LLM.
- Add rate limiting / abuse detection for public endpoints.

## 🧪 Testing
Run the included basic tests (no external API calls):
```bash
pytest -q
```

Included:
- `test_chunking.py`: Validates deterministic chunking & metadata.
- `test_satisfaction.py`: Validates satisfaction trend logic.

Recommended future tests (not yet implemented):
- Mocked tests for sentiment & escalation (patch OpenAI responses)
- Retrieval integration test with ephemeral Pinecone index
- Consistency regression test for evaluation scoring

## 🛠 Troubleshooting
| Issue | Cause | Resolution |
|-------|-------|-----------|
| Pinecone init delay | Fresh index creation | Wait 10–20s first run |
| No results | Index empty | Reload KB in sidebar |
| OpenAI rate limit | High request volume | Backoff & batching |
| Streamlit reruns | Exceptions in init | Check sidebar error messages |

## 🔮 Future Enhancements

- **Multi-language Support**: Sentiment analysis and responses in multiple languages (prompt-based adaptation + future model selection)
- **Voice Integration**: Speech-to-text and text-to-speech capabilities
- **Advanced Analytics**: Predictive customer satisfaction modeling & deeper automated tone adaptation based on rolling satisfaction
- **Integration APIs**: REST endpoints for external system integration
- **Custom Training**: Fine-tuned models for domain-specific responses

## 📄 License

This project is created for educational purposes as part of a RAG system assignment.

## 🤝 Contributing

This is an academic project, but suggestions and improvements are welcome through issues and pull requests.

---

**Built with ❤️ using OpenAI GPT-4o, Pinecone, and Streamlit**