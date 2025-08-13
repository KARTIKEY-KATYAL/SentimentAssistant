import json
import time
from typing import List, Dict, Any
from openai import OpenAI
import streamlit as st
import config

class EscalationPredictor:
    def __init__(self):
        """Initialize escalation predictor"""
        self.openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
        
        # Escalation risk thresholds
        self.risk_levels = {
            (0.0, 0.3): "low",
            (0.3, 0.6): "medium", 
            (0.6, 0.8): "high",
            (0.8, 1.0): "critical"
        }
        
        # Pattern indicators
        self.escalation_patterns = {
            "repeated_issues": 0.7,
            "multiple_channels": 0.6,
            "time_pressure": 0.5,
            "previous_escalations": 0.8,
            "sentiment_deterioration": 0.6,
            "unresolved_complaints": 0.7
        }
    
    def predict_escalation(self, current_message: str, conversation_history: List[Dict]) -> float:
        """Predict escalation risk based on current message and conversation history"""
        try:
            # Extract features from conversation
            features = self._extract_conversation_features(conversation_history)
            
            # Analyze current message
            message_features = self._analyze_current_message(current_message)
            
            # Combine features
            all_features = {**features, **message_features}
            
            # Use AI to predict escalation risk
            escalation_risk = self._ai_predict_escalation(current_message, all_features, conversation_history)
            
            return max(0.0, min(1.0, escalation_risk))
            
        except Exception as e:
            st.error(f"Escalation prediction failed: {str(e)}")
            return 0.5  # Default medium risk
    
    def _extract_conversation_features(self, conversation_history: List[Dict]) -> Dict[str, Any]:
        """Extract features from conversation history"""
        if not conversation_history:
            return {}
        
        customer_messages = [msg for msg in conversation_history if msg['sender'] == 'customer']
        agent_messages = [msg for msg in conversation_history if msg['sender'] == 'agent']
        
        features = {
            "message_count": len(customer_messages),
            "conversation_length": len(conversation_history),
            "response_ratio": len(agent_messages) / max(len(customer_messages), 1),
            "conversation_duration": 0,
            "sentiment_trend": 0.5,
            "repeated_issues": False,
            "unresolved_count": 0
        }
        
        # Calculate conversation duration
        if conversation_history:
            start_time = conversation_history[0].get('timestamp', time.time())
            end_time = conversation_history[-1].get('timestamp', time.time())
            features["conversation_duration"] = end_time - start_time
        
        # Analyze sentiment trend
        sentiment_scores = [
            msg.get('sentiment_score', 0.5) 
            for msg in customer_messages 
            if 'sentiment_score' in msg
        ]
        
        if len(sentiment_scores) >= 2:
            # Calculate sentiment trend (negative = deteriorating)
            recent_sentiment = sum(sentiment_scores[-3:]) / min(3, len(sentiment_scores))
            earlier_sentiment = sum(sentiment_scores[:3]) / min(3, len(sentiment_scores))
            features["sentiment_trend"] = recent_sentiment - earlier_sentiment
            features["current_sentiment"] = sentiment_scores[-1]
            features["min_sentiment"] = min(sentiment_scores)
        
        # Check for repeated issues (simplified)
        customer_content = [msg['content'].lower() for msg in customer_messages]
        if len(customer_content) > 2:
            # Look for repeated keywords
            all_words = ' '.join(customer_content).split()
            word_freq = {}
            for word in all_words:
                if len(word) > 4:  # Only consider longer words
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # If any word appears more than twice, consider it repeated
            features["repeated_issues"] = any(count > 2 for count in word_freq.values())
        
        return features
    
    def _analyze_current_message(self, message: str) -> Dict[str, Any]:
        """Analyze current message for escalation indicators"""
        escalation_keywords = [
            "manager", "supervisor", "complaint", "unacceptable",
            "cancel", "refund", "lawsuit", "terrible", "awful",
            "frustrated", "angry", "disappointed", "urgent"
        ]
        
        message_lower = message.lower()
        
        features = {
            "message_length": len(message),
            "escalation_keywords": sum(1 for keyword in escalation_keywords if keyword in message_lower),
            "caps_ratio": sum(1 for c in message if c.isupper()) / max(len(message), 1),
            "question_marks": message.count('?'),
            "exclamation_marks": message.count('!'),
            "urgency_indicators": 0
        }
        
        # Check for urgency indicators
        urgency_words = ["urgent", "asap", "immediately", "now", "today", "emergency"]
        features["urgency_indicators"] = sum(1 for word in urgency_words if word in message_lower)
        
        return features
    
    def _ai_predict_escalation(self, message: str, features: Dict, conversation_history: List[Dict]) -> float:
        """Use AI to predict escalation risk based on features and context"""
        try:
            # Prepare conversation context
            recent_messages = conversation_history[-5:] if conversation_history else []
            context = "\n".join([
                f"{msg['sender']}: {msg['content']}" 
                for msg in recent_messages
            ])
            
            prompt = f"""
            Analyze the escalation risk for this customer support conversation.
            
            Current customer message: "{message}"
            
            Recent conversation context:
            {context}
            
            Conversation features:
            - Message count: {features.get('message_count', 0)}
            - Conversation duration: {features.get('conversation_duration', 0):.0f} seconds
            - Sentiment trend: {features.get('sentiment_trend', 0):.2f} (negative = deteriorating)
            - Current sentiment: {features.get('current_sentiment', 0.5):.2f}
            - Escalation keywords found: {features.get('escalation_keywords', 0)}
            - Urgency indicators: {features.get('urgency_indicators', 0)}
            - Caps usage ratio: {features.get('caps_ratio', 0):.2f}
            
            Consider these factors:
            1. Sentiment deterioration over time
            2. Use of escalation keywords
            3. Urgency and frustration indicators
            4. Length and complexity of the issue
            5. Customer communication patterns
            
            Provide an escalation risk score from 0.0 (no risk) to 1.0 (immediate escalation likely).
            
            Respond with JSON:
            {{
                "escalation_risk": number,
                "confidence": number,
                "primary_factors": ["factor1", "factor2", "factor3"],
                "recommendation": "action_recommendation"
            }}
            """
            
            response = self.openai_client.chat.completions.create(
                model=config.CHAT_MODEL,  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in customer service escalation prediction. "
                        "Analyze conversations to predict when customers might escalate issues."
                    },
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            
            result = json.loads(response.choices[0].message.content)
            return result.get("escalation_risk", 0.5)
            
        except Exception as e:
            st.error(f"AI escalation prediction failed: {str(e)}")
            
            # Fallback: rule-based prediction
            return self._rule_based_prediction(features)
    
    def _rule_based_prediction(self, features: Dict) -> float:
        """Fallback rule-based escalation prediction"""
        risk_score = 0.0
        
        # Sentiment-based risk
        current_sentiment = features.get('current_sentiment', 0.5)
        if current_sentiment < 0.3:
            risk_score += 0.4
        elif current_sentiment < 0.5:
            risk_score += 0.2
        
        # Keyword-based risk
        escalation_keywords = features.get('escalation_keywords', 0)
        risk_score += min(0.3, escalation_keywords * 0.1)
        
        # Urgency-based risk
        urgency_indicators = features.get('urgency_indicators', 0)
        risk_score += min(0.2, urgency_indicators * 0.1)
        
        # Conversation length risk
        message_count = features.get('message_count', 0)
        if message_count > 10:
            risk_score += 0.3
        elif message_count > 5:
            risk_score += 0.1
        
        # Sentiment trend risk
        sentiment_trend = features.get('sentiment_trend', 0)
        if sentiment_trend < -0.2:  # Deteriorating sentiment
            risk_score += 0.3
        
        return min(1.0, risk_score)
    
    def get_escalation_recommendations(self, risk_score: float, features: Dict) -> Dict[str, Any]:
        """Get recommendations based on escalation risk"""
        risk_level = self._get_risk_level(risk_score)
        
        recommendations = {
            "low": {
                "actions": ["Continue standard support", "Monitor sentiment"],
                "priority": "normal",
                "response_time": "within 24 hours"
            },
            "medium": {
                "actions": ["Prioritize response", "Consider proactive follow-up", "Manager review"],
                "priority": "high", 
                "response_time": "within 4 hours"
            },
            "high": {
                "actions": ["Immediate supervisor involvement", "Expedited resolution", "Proactive communication"],
                "priority": "urgent",
                "response_time": "within 1 hour"
            },
            "critical": {
                "actions": ["Immediate escalation", "Manager intervention", "Priority handling"],
                "priority": "critical",
                "response_time": "within 15 minutes"
            }
        }
        
        return {
            "risk_level": risk_level,
            "risk_score": risk_score,
            **recommendations[risk_level]
        }
    
    def _get_risk_level(self, score: float) -> str:
        """Get risk level based on score"""
        for (min_val, max_val), level in self.risk_levels.items():
            if min_val <= score < max_val:
                return level
        return "critical" if score >= 0.8 else "low"
