# Overview

This is an AI-powered customer support system built with Streamlit that provides intelligent, context-aware responses to customer inquiries. The system combines retrieval-augmented generation (RAG) with sentiment analysis, escalation prediction, and response personalization to deliver empathetic customer service. It uses OpenAI's GPT models for natural language processing, Pinecone for vector similarity search, and includes comprehensive evaluation metrics to continuously improve response quality.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Core Components Architecture

The system follows a modular microservices-inspired architecture with clear separation of concerns:

**Frontend Layer**: Streamlit-based web interface that handles user interactions and displays conversation history, analytics dashboards, and system metrics.

**AI Processing Pipeline**: Multi-stage processing pipeline that includes:
- Vector Store module for knowledge retrieval using Pinecone and OpenAI embeddings
- Sentiment Analysis module for emotional state detection and urgency assessment
- Escalation Predictor module for risk assessment and proactive escalation management
- Response Generator module for contextual, tone-appropriate response creation

**Knowledge Management**: Knowledge base system with JSON-based storage, document processing, and vector indexing for efficient retrieval.

**Evaluation System**: RAGAS-inspired evaluation framework that measures context precision, faithfulness, answer relevancy, and other quality metrics.

## Data Flow Design

The system processes customer messages through a sequential pipeline:
1. Message ingestion and sentiment analysis
2. Vector similarity search for relevant knowledge base articles
3. Escalation risk prediction based on conversation history and sentiment
4. Contextual response generation with appropriate tone adjustment
5. Response evaluation and quality scoring

## State Management

Session state management handles conversation persistence, customer identification, satisfaction tracking, and escalation alerts. This ensures continuity across user interactions and enables analytics.

## Configuration Management

Centralized configuration system manages API keys, model parameters, sentiment thresholds, and response settings, allowing for easy tuning without code changes.

# External Dependencies

## AI Services
- **OpenAI API**: Powers natural language processing, embeddings (text-embedding-3-small), and chat completions (gpt-4o)
- **Pinecone**: Vector database service for semantic search and knowledge base indexing

## Framework Dependencies
- **Streamlit**: Web application framework for the user interface and real-time interactions
- **Plotly**: Data visualization library for analytics dashboards and metrics display
- **Pandas**: Data manipulation and analysis for conversation history and metrics processing

## Development Tools
- **NumPy**: Numerical computing for evaluation metrics and data processing
- **JSON**: Data serialization for knowledge base storage and API communications

## Environment Configuration
The system expects environment variables for API keys and service configuration, with fallback defaults for development environments.