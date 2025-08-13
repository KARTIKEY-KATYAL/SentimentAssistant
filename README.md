# Customer Support RAG with Sentiment Analysis

A comprehensive AI-powered customer support system that combines Retrieval-Augmented Generation (RAG) with real-time sentiment analysis, escalation prediction, and empathetic response generation.

## 🚀 Live Demo

**Deployed Application:** [View Live Demo](https://your-replit-url.replit.app)

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
- **Customer Satisfaction Tracking**: Analytics dashboard with conversation metrics

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

1. **Vector Store Module**: Manages Pinecone integration and semantic search
2. **Sentiment Analyzer**: Real-time emotion and sentiment detection
3. **Escalation Predictor**: Risk assessment and escalation alerts
4. **Response Generator**: Context-aware, empathetic response creation
5. **Knowledge Processor**: Article management and indexing
6. **RAG Evaluator**: Quality metrics and performance assessment

## 🚀 Quick Start

### Prerequisites
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))
- Pinecone API key ([Get one here](https://www.pinecone.io/))

### Environment Setup
Set these environment variables in Replit Secrets:
```bash
OPENAI_API_KEY=your_openai_key_here
PINECONE_API_KEY=your_pinecone_key_here
```

### Installation
```bash
# Install dependencies
pip install streamlit openai pinecone plotly pandas

# Run the application
streamlit run app.py --server.port 5000
```

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
├── app.py                      # Main Streamlit application
├── config.py                   # Configuration settings
├── modules/
│   ├── vector_store.py         # Pinecone integration
│   ├── sentiment_analysis.py   # Emotion detection
│   ├── escalation_predictor.py # Risk assessment
│   ├── response_generator.py   # AI response creation
│   ├── knowledge_processor.py  # Knowledge base management
│   └── evaluation.py           # RAGAS metrics
├── utils/
│   └── helpers.py             # Utility functions
├── data/
│   └── sample_knowledge_base.json # Sample help articles
└── .streamlit/
    └── config.toml            # Streamlit configuration
```

## 🚀 Deployment

The application is deployed on Replit with the following configuration:

- **Port**: 5000
- **Server**: 0.0.0.0 (accessible externally)
- **Auto-restart**: Enabled for continuous availability

### Deployment URL
Access the live application at: `https://your-replit-url.replit.app`

## 🔮 Future Enhancements

- **Multi-language Support**: Sentiment analysis and responses in multiple languages
- **Voice Integration**: Speech-to-text and text-to-speech capabilities
- **Advanced Analytics**: Predictive customer satisfaction modeling
- **Integration APIs**: REST endpoints for external system integration
- **Custom Training**: Fine-tuned models for domain-specific responses

## 📄 License

This project is created for educational purposes as part of a RAG system assignment.

## 🤝 Contributing

This is an academic project, but suggestions and improvements are welcome through issues and pull requests.

---

**Built with ❤️ using OpenAI GPT-4o, Pinecone, and Streamlit**