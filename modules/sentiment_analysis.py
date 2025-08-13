import openai
from openai import OpenAI
import json
import streamlit as st
import time
from typing import Dict, Any
import config

class SentimentAnalyzer:
    def __init__(self):
        """Initialize sentiment analyzer with OpenAI client"""
        self.openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
        
        # Sentiment mapping
        self.sentiment_labels = {
            (0.0, 0.2): "very_negative",
            (0.2, 0.4): "negative", 
            (0.4, 0.6): "neutral",
            (0.6, 0.8): "positive",
            (0.8, 1.0): "very_positive"
        }
        
        # Emotion categories
        self.emotions = [
            "anger", "frustration", "disappointment", "confusion",
            "neutral", "satisfaction", "happiness", "excitement"
        ]
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment and emotions in customer text"""
        try:
            prompt = f"""
            Analyze the sentiment and emotional state of the following customer message.
            Provide a comprehensive analysis including:
            
            1. Sentiment score (0.0 = very negative, 1.0 = very positive)
            2. Confidence level (0.0 to 1.0)
            3. Primary emotion from: {', '.join(self.emotions)}
            4. Urgency level (0.0 = low, 1.0 = urgent)
            5. Key indicators that led to this analysis
            
            Customer message: "{text}"
            
            Respond with JSON in this exact format:
            {{
                "sentiment_score": number,
                "confidence": number,
                "primary_emotion": "emotion_name",
                "urgency": number,
                "indicators": ["indicator1", "indicator2", "indicator3"]
            }}
            """
            
            response = self.openai_client.chat.completions.create(
                model=config.CHAT_MODEL,  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert sentiment and emotion analysis AI. "
                        "Provide accurate, nuanced analysis of customer communications."
                    },
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Validate and normalize results
            sentiment_score = max(0.0, min(1.0, result.get("sentiment_score", 0.5)))
            confidence = max(0.0, min(1.0, result.get("confidence", 0.5)))
            urgency = max(0.0, min(1.0, result.get("urgency", 0.5)))
            
            # Get sentiment label
            sentiment_label = self._get_sentiment_label(sentiment_score)
            
            return {
                "sentiment_score": sentiment_score,
                "sentiment_label": sentiment_label,
                "confidence": confidence,
                "primary_emotion": result.get("primary_emotion", "neutral"),
                "urgency": urgency,
                "indicators": result.get("indicators", []),
                "analysis_timestamp": time.time()
            }
            
        except Exception as e:
            st.error(f"Sentiment analysis failed: {str(e)}")
            # Return neutral sentiment as fallback
            return {
                "sentiment_score": 0.5,
                "sentiment_label": "neutral",
                "confidence": 0.1,
                "primary_emotion": "neutral", 
                "urgency": 0.5,
                "indicators": ["analysis_failed"],
                "analysis_timestamp": time.time()
            }
    
    def _get_sentiment_label(self, score: float) -> str:
        """Get sentiment label based on score"""
        for (min_val, max_val), label in self.sentiment_labels.items():
            if min_val <= score < max_val:
                return label
        return "very_positive" if score >= 0.8 else "very_negative"
    
    def analyze_conversation_trend(self, conversation_history: list) -> Dict[str, Any]:
        """Analyze sentiment trend across conversation"""
        try:
            customer_messages = [
                msg for msg in conversation_history 
                if msg['sender'] == 'customer' and 'sentiment_score' in msg
            ]
            
            if len(customer_messages) < 2:
                return {"trend": "insufficient_data"}
            
            scores = [msg['sentiment_score'] for msg in customer_messages]
            
            # Calculate trend
            if len(scores) >= 3:
                recent_avg = sum(scores[-3:]) / 3
                earlier_avg = sum(scores[:-3]) / len(scores[:-3]) if len(scores) > 3 else scores[0]
                trend_direction = recent_avg - earlier_avg
            else:
                trend_direction = scores[-1] - scores[0]
            
            # Determine trend category
            if trend_direction > 0.1:
                trend = "improving"
            elif trend_direction < -0.1:
                trend = "deteriorating"
            else:
                trend = "stable"
            
            return {
                "trend": trend,
                "trend_strength": abs(trend_direction),
                "current_score": scores[-1],
                "average_score": sum(scores) / len(scores),
                "score_range": max(scores) - min(scores),
                "message_count": len(customer_messages)
            }
            
        except Exception as e:
            st.error(f"Conversation trend analysis failed: {str(e)}")
            return {"trend": "analysis_failed"}
    
    def detect_escalation_triggers(self, text: str) -> Dict[str, Any]:
        """Detect specific escalation triggers in customer message"""
        try:
            escalation_keywords = [
                "manager", "supervisor", "complaint", "unacceptable", 
                "terrible", "awful", "worst", "cancel", "refund",
                "lawsuit", "attorney", "fraud", "scam", "useless"
            ]
            
            # Check for escalation keywords
            text_lower = text.lower()
            triggers_found = [keyword for keyword in escalation_keywords if keyword in text_lower]
            
            # Analyze escalation intent with AI
            if triggers_found:
                prompt = f"""
                Analyze this customer message for escalation intent and severity.
                Message: "{text}"
                
                Keywords detected: {triggers_found}
                
                Rate the escalation likelihood (0.0 to 1.0) and provide reasoning.
                Respond with JSON:
                {{
                    "escalation_likelihood": number,
                    "severity": "low|medium|high|critical",
                    "reasoning": "explanation"
                }}
                """
                
                response = self.openai_client.chat.completions.create(
                    model=config.CHAT_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.2
                )
                
                ai_result = json.loads(response.choices[0].message.content)
                
                return {
                    "triggers_found": triggers_found,
                    "escalation_likelihood": ai_result.get("escalation_likelihood", 0.5),
                    "severity": ai_result.get("severity", "medium"),
                    "reasoning": ai_result.get("reasoning", ""),
                    "keyword_count": len(triggers_found)
                }
            else:
                return {
                    "triggers_found": [],
                    "escalation_likelihood": 0.0,
                    "severity": "low",
                    "reasoning": "No escalation keywords detected",
                    "keyword_count": 0
                }
                
        except Exception as e:
            st.error(f"Escalation trigger detection failed: {str(e)}")
            return {
                "triggers_found": [],
                "escalation_likelihood": 0.0,
                "severity": "unknown",
                "reasoning": "Analysis failed",
                "keyword_count": 0
            }
    
    def is_healthy(self) -> bool:
        """Check if sentiment analyzer is working properly"""
        try:
            # Test with simple message
            test_result = self.analyze_sentiment("Hello, I need help with my account.")
            return "sentiment_score" in test_result
        except:
            return False
