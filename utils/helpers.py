import time
from datetime import datetime
from typing import Any, Dict, List
import streamlit as st

def format_timestamp(timestamp: float) -> str:
    """Format timestamp to readable string"""
    try:
        dt = datetime.fromtimestamp(timestamp)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return "Unknown"

def calculate_response_time(start_time: float, end_time: float) -> float:
    """Calculate response time in seconds"""
    return max(0.0, end_time - start_time)

def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to specified length"""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."

def format_conversation_for_display(conversation_history: List[Dict]) -> List[Dict]:
    """Format conversation history for display"""
    formatted = []
    
    for msg in conversation_history:
        formatted_msg = {
            "sender": msg["sender"],
            "content": msg["content"],
            "timestamp": format_timestamp(msg.get("timestamp", time.time())),
            "display_time": datetime.fromtimestamp(msg.get("timestamp", time.time())).strftime("%H:%M")
        }
        
        # Add metadata for customer messages
        if msg["sender"] == "customer":
            if "sentiment_score" in msg:
                formatted_msg["sentiment"] = {
                    "score": msg["sentiment_score"],
                    "label": msg.get("sentiment_label", "neutral"),
                    "emotion": msg.get("primary_emotion", "neutral")
                }
            
            if "escalation_risk" in msg:
                formatted_msg["escalation_risk"] = msg["escalation_risk"]
        
        # Add metadata for agent messages
        elif msg["sender"] == "agent":
            if "response_time" in msg:
                formatted_msg["response_time"] = f"{msg['response_time']:.2f}s"
            
            if "tone" in msg:
                formatted_msg["tone"] = msg["tone"]
        
        formatted.append(formatted_msg)
    
    return formatted

def get_sentiment_emoji(sentiment_score: float) -> str:
    """Get emoji based on sentiment score"""
    if sentiment_score < 0.2:
        return "😡"
    elif sentiment_score < 0.4:
        return "😞"
    elif sentiment_score < 0.6:
        return "😐"
    elif sentiment_score < 0.8:
        return "🙂"
    else:
        return "😊"

def get_urgency_emoji(urgency_score: float) -> str:
    """Get emoji based on urgency score"""
    if urgency_score < 0.3:
        return "🟢"  # Low urgency
    elif urgency_score < 0.6:
        return "🟡"  # Medium urgency
    else:
        return "🔴"  # High urgency

def format_metric_delta(current: float, previous: float) -> str:
    """Format metric delta for display"""
    if previous == 0:
        return "N/A"
    
    delta = ((current - previous) / previous) * 100
    sign = "+" if delta > 0 else ""
    return f"{sign}{delta:.1f}%"

def validate_api_keys() -> Dict[str, bool]:
    """Validate that required API keys are available"""
    import config
    
    validation = {}
    
    # Check OpenAI API key
    try:
        validation["openai"] = bool(config.OPENAI_API_KEY and config.OPENAI_API_KEY != "default_openai_key")
    except:
        validation["openai"] = False
    
    # Check Pinecone API key
    try:
        validation["pinecone"] = bool(config.PINECONE_API_KEY and config.PINECONE_API_KEY != "default_pinecone_key")
    except:
        validation["pinecone"] = False
    
    validation["all_valid"] = all(validation.values())
    
    return validation

def create_download_link(data: str, filename: str, link_text: str) -> str:
    """Create download link for data"""
    import base64
    
    b64 = base64.b64encode(data.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero"""
    try:
        return numerator / denominator if denominator != 0 else default
    except:
        return default

def calculate_conversation_metrics(conversation_history: List[Dict]) -> Dict[str, Any]:
    """Calculate comprehensive conversation metrics"""
    if not conversation_history:
        return {}
    
    customer_messages = [msg for msg in conversation_history if msg["sender"] == "customer"]
    agent_messages = [msg for msg in conversation_history if msg["sender"] == "agent"]
    
    metrics = {
        "total_messages": len(conversation_history),
        "customer_messages": len(customer_messages),
        "agent_messages": len(agent_messages),
        "conversation_duration": 0,
        "avg_response_time": 0,
        "sentiment_trend": "neutral",
        "escalation_count": 0
    }
    
    # Calculate conversation duration
    if conversation_history:
        start_time = min(msg.get("timestamp", time.time()) for msg in conversation_history)
        end_time = max(msg.get("timestamp", time.time()) for msg in conversation_history)
        metrics["conversation_duration"] = end_time - start_time
    
    # Calculate average response time
    response_times = [msg.get("response_time", 0) for msg in agent_messages if "response_time" in msg]
    if response_times:
        metrics["avg_response_time"] = sum(response_times) / len(response_times)
    
    # Analyze sentiment trend
    sentiment_scores = [msg.get("sentiment_score", 0.5) for msg in customer_messages if "sentiment_score" in msg]
    if len(sentiment_scores) >= 2:
        if sentiment_scores[-1] > sentiment_scores[0] + 0.1:
            metrics["sentiment_trend"] = "improving"
        elif sentiment_scores[-1] < sentiment_scores[0] - 0.1:
            metrics["sentiment_trend"] = "deteriorating"
        else:
            metrics["sentiment_trend"] = "stable"
    
    # Count escalation alerts
    metrics["escalation_count"] = sum(1 for msg in customer_messages if msg.get("escalation_risk", 0) > 0.6)
    
    return metrics

def format_duration(seconds: float) -> str:
    """Format duration in seconds to human readable string"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

@st.cache_data
def get_color_scheme() -> Dict[str, str]:
    """Get consistent color scheme for the application"""
    return {
        "primary": "#1f77b4",
        "secondary": "#ff7f0e", 
        "success": "#2ca02c",
        "warning": "#d62728",
        "info": "#17becf",
        "light": "#f8f9fa",
        "dark": "#212529",
        "sentiment_positive": "#28a745",
        "sentiment_neutral": "#ffc107",
        "sentiment_negative": "#dc3545"
    }
