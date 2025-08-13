import streamlit as st
import time
import json
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from modules.vector_store import VectorStore
from modules.sentiment_analysis import SentimentAnalyzer
from modules.escalation_predictor import EscalationPredictor
from modules.response_generator import ResponseGenerator
from modules.knowledge_processor import KnowledgeProcessor
from modules.evaluation import RAGEvaluator
from utils.helpers import format_timestamp, calculate_response_time
import config

# Initialize session state
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'customer_id' not in st.session_state:
    st.session_state.customer_id = f"customer_{int(time.time())}"
if 'satisfaction_scores' not in st.session_state:
    st.session_state.satisfaction_scores = []
if 'escalation_alerts' not in st.session_state:
    st.session_state.escalation_alerts = []

# Initialize components
@st.cache_resource
def initialize_components():
    """Initialize all system components"""
    try:
        vector_store = VectorStore()
        sentiment_analyzer = SentimentAnalyzer()
        escalation_predictor = EscalationPredictor()
        response_generator = ResponseGenerator()
        knowledge_processor = KnowledgeProcessor()
        evaluator = RAGEvaluator()
        
        return {
            'vector_store': vector_store,
            'sentiment_analyzer': sentiment_analyzer,
            'escalation_predictor': escalation_predictor,
            'response_generator': response_generator,
            'knowledge_processor': knowledge_processor,
            'evaluator': evaluator
        }
    except Exception as e:
        st.error(f"Failed to initialize components: {str(e)}")
        return None

def main():
    st.set_page_config(
        page_title="Customer Support RAG System",
        page_icon="🎧",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🎧 Customer Support RAG System")
    st.subheader("AI-Powered Support with Sentiment Analysis & Escalation Prediction")
    
    # Initialize components
    components = initialize_components()
    if not components:
        st.error("System initialization failed. Please check your API keys and try again.")
        return
    
    # Sidebar for system status and metrics
    with st.sidebar:
        st.header("📊 System Status")
        
        # System health check
        with st.spinner("Checking system health..."):
            health_status = check_system_health(components)
            
        if health_status['all_healthy']:
            st.success("✅ All systems operational")
        else:
            st.warning("⚠️ Some systems need attention")
            for service, status in health_status.items():
                if service != 'all_healthy' and not status:
                    st.error(f"❌ {service}")
        
        st.divider()
        
        # Customer information
        st.header("👤 Customer Info")
        st.write(f"**Customer ID:** {st.session_state.customer_id}")
        st.write(f"**Session Started:** {format_timestamp(time.time())}")
        
        # Conversation metrics
        if st.session_state.conversation_history:
            st.metric("Messages", len(st.session_state.conversation_history))
            
            # Calculate average sentiment
            sentiments = [msg.get('sentiment_score', 0.5) for msg in st.session_state.conversation_history if msg['sender'] == 'customer']
            if sentiments:
                avg_sentiment = sum(sentiments) / len(sentiments)
                st.metric("Avg Sentiment", f"{avg_sentiment:.2f}")
        
        st.divider()
        
        # Knowledge base management
        st.header("📚 Knowledge Base")
        if st.button("🔄 Reload Knowledge Base"):
            with st.spinner("Reloading knowledge base..."):
                success = components['knowledge_processor'].load_knowledge_base()
                if success:
                    st.success("Knowledge base reloaded successfully!")
                else:
                    st.error("Failed to reload knowledge base")
        
        # Clear conversation
        if st.button("🗑️ Clear Conversation"):
            st.session_state.conversation_history = []
            st.session_state.escalation_alerts = []
            st.rerun()
    
    # Main chat interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Customer Support Chat")
        
        # Display conversation history
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.conversation_history:
                if msg['sender'] == 'customer':
                    with st.chat_message("user"):
                        st.write(msg['content'])
                        
                        # Show sentiment and escalation info
                        if 'sentiment_score' in msg:
                            sentiment_color = get_sentiment_color(msg['sentiment_score'])
                            st.caption(f"Sentiment: {msg['sentiment_label']} ({msg['sentiment_score']:.2f})")
                            
                        if msg.get('escalation_risk', 0) > config.ESCALATION_THRESHOLD:
                            st.warning(f"⚠️ Escalation Risk: {msg['escalation_risk']:.2f}")
                
                elif msg['sender'] == 'agent':
                    with st.chat_message("assistant"):
                        st.write(msg['content'])
                        
                        # Show response metadata
                        if 'response_time' in msg:
                            st.caption(f"Response time: {msg['response_time']:.2f}s | Tone: {msg.get('tone', 'neutral')}")
        
        # Customer input
        customer_message = st.chat_input("Type your message here...")
        
        if customer_message:
            process_customer_message(customer_message, components)
    
    with col2:
        st.header("📈 Analytics Dashboard")
        
        # Escalation alerts
        if st.session_state.escalation_alerts:
            st.subheader("🚨 Escalation Alerts")
            for alert in st.session_state.escalation_alerts[-3:]:  # Show last 3 alerts
                st.error(f"⚠️ {alert['message']} (Risk: {alert['risk']:.2f})")
        
        # Sentiment trend
        if len(st.session_state.conversation_history) > 1:
            st.subheader("😊 Sentiment Trend")
            sentiment_data = []
            for i, msg in enumerate(st.session_state.conversation_history):
                if msg['sender'] == 'customer' and 'sentiment_score' in msg:
                    sentiment_data.append({
                        'Message': i + 1,
                        'Sentiment': msg['sentiment_score'],
                        'Label': msg['sentiment_label']
                    })
            
            if sentiment_data:
                df = pd.DataFrame(sentiment_data)
                fig = px.line(df, x='Message', y='Sentiment', 
                            title="Customer Sentiment Over Time",
                            range_y=[0, 1])
                fig.add_hline(y=0.5, line_dash="dash", line_color="gray", 
                            annotation_text="Neutral")
                fig.add_hline(y=config.ESCALATION_THRESHOLD, line_dash="dash", 
                            line_color="red", annotation_text="Escalation Threshold")
                st.plotly_chart(fig, use_container_width=True)
        
        # System performance metrics
        st.subheader("⚡ Performance Metrics")
        if st.session_state.conversation_history:
            response_times = [msg.get('response_time', 0) for msg in st.session_state.conversation_history if msg['sender'] == 'agent']
            if response_times:
                avg_response_time = sum(response_times) / len(response_times)
                st.metric("Avg Response Time", f"{avg_response_time:.2f}s")
                
                # Response time distribution
                fig = px.histogram(response_times, title="Response Time Distribution")
                st.plotly_chart(fig, use_container_width=True)

def check_system_health(components):
    """Check health status of all system components"""
    health_status = {}
    
    try:
        # Check vector store
        health_status['vector_store'] = components['vector_store'].is_healthy()
    except:
        health_status['vector_store'] = False
    
    try:
        # Check sentiment analyzer
        health_status['sentiment_analyzer'] = components['sentiment_analyzer'].is_healthy()
    except:
        health_status['sentiment_analyzer'] = False
    
    try:
        # Check response generator
        health_status['response_generator'] = components['response_generator'].is_healthy()
    except:
        health_status['response_generator'] = False
    
    health_status['all_healthy'] = all(health_status.values())
    return health_status

def get_sentiment_color(sentiment_score):
    """Get color based on sentiment score"""
    if sentiment_score < 0.3:
        return "red"
    elif sentiment_score < 0.7:
        return "orange"
    else:
        return "green"

def process_customer_message(message, components):
    """Process incoming customer message and generate response"""
    start_time = time.time()
    
    # Add customer message to history
    customer_msg = {
        'sender': 'customer',
        'content': message,
        'timestamp': time.time()
    }
    
    try:
        # Analyze sentiment
        sentiment_result = components['sentiment_analyzer'].analyze_sentiment(message)
        customer_msg.update(sentiment_result)
        
        # Predict escalation risk
        escalation_risk = components['escalation_predictor'].predict_escalation(
            message, st.session_state.conversation_history
        )
        customer_msg['escalation_risk'] = escalation_risk
        
        # Check for escalation
        if escalation_risk > config.ESCALATION_THRESHOLD:
            alert = {
                'message': f"High escalation risk detected in customer message",
                'risk': escalation_risk,
                'timestamp': time.time()
            }
            st.session_state.escalation_alerts.append(alert)
        
        st.session_state.conversation_history.append(customer_msg)
        
        # Generate response
        with st.spinner("Generating response..."):
            # Retrieve relevant knowledge
            retrieved_docs = components['vector_store'].similarity_search(message, k=3)
            
            # Generate empathetic response
            response_data = components['response_generator'].generate_response(
                message, 
                retrieved_docs, 
                st.session_state.conversation_history,
                sentiment_result
            )
            
            response_time = time.time() - start_time
            
            # Add agent response to history
            agent_msg = {
                'sender': 'agent',
                'content': response_data['response'],
                'timestamp': time.time(),
                'response_time': response_time,
                'tone': response_data.get('tone', 'neutral'),
                'retrieved_docs': len(retrieved_docs)
            }
            
            st.session_state.conversation_history.append(agent_msg)
            
            # Evaluate response quality
            try:
                evaluation_metrics = components['evaluator'].evaluate_response(
                    message, response_data['response'], retrieved_docs
                )
                agent_msg['evaluation'] = evaluation_metrics
            except Exception as e:
                st.warning(f"Response evaluation failed: {str(e)}")
        
        st.rerun()
        
    except Exception as e:
        st.error(f"Error processing message: {str(e)}")

if __name__ == "__main__":
    main()
