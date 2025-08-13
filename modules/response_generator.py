import json
import time
from typing import List, Dict, Any
from openai import OpenAI
import streamlit as st
import config

class ResponseGenerator:
    def __init__(self):
        """Initialize response generator with OpenAI client"""
        self.openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
        
        # Response tone styles
        self.tone_styles = {
            "empathetic": "warm, understanding, and supportive",
            "professional": "formal, courteous, and business-like",
            "apologetic": "sorry, acknowledging fault, and remedial",
            "reassuring": "confident, calming, and solution-focused",
            "urgent": "immediate, direct, and action-oriented"
        }
        
        # Escalation response templates
        self.escalation_responses = {
            "high": "I understand your frustration, and I want to ensure we resolve this quickly for you.",
            "critical": "I sincerely apologize for the inconvenience you've experienced. Let me connect you with a supervisor immediately."
        }
    
    def generate_response(self, customer_message: str, retrieved_docs: List[Dict], 
                         conversation_history: List[Dict], sentiment_data: Dict, satisfaction_avg: float | None = None) -> Dict[str, Any]:
        """Generate empathetic and contextually appropriate response"""
        try:
            # Determine appropriate tone based on sentiment and escalation risk
            tone = self._determine_response_tone(sentiment_data, conversation_history, satisfaction_avg)
            
            # Prepare context from retrieved documents
            context = self._prepare_context(retrieved_docs)
            
            # Generate response using AI
            response = self._generate_ai_response(
                customer_message, context, conversation_history, tone, sentiment_data
            )
            
            return {
                "response": response,
                "tone": tone,
                "context_used": len(retrieved_docs),
                "generation_timestamp": time.time()
            }
            
        except Exception as e:
            st.error(f"Response generation failed: {str(e)}")
            return {
                "response": "I apologize, but I'm having trouble processing your request right now. Let me connect you with a human agent who can assist you better.",
                "tone": "apologetic",
                "context_used": 0,
                "generation_timestamp": time.time()
            }
    
    def _determine_response_tone(self, sentiment_data: Dict, conversation_history: List[Dict], satisfaction_avg: float | None) -> str:
        """Determine appropriate response tone based on customer sentiment, context, and satisfaction trend."""
        sentiment_score = sentiment_data.get('sentiment_score', 0.5)
        primary_emotion = sentiment_data.get('primary_emotion', 'neutral')
        urgency = sentiment_data.get('urgency', 0.5)
        # Satisfaction influence: if low average rating (<3) shift to empathetic/apologetic
        if satisfaction_avg is not None and satisfaction_avg < 3:
            if sentiment_score < 0.4:
                return "apologetic"
            return "empathetic"
        if satisfaction_avg is not None and satisfaction_avg > 4.3 and sentiment_score > 0.6:
            return "reassuring"
        
        # High urgency situations
        if urgency > 0.8:
            return "urgent"
        
        # Very negative sentiment
        if sentiment_score < 0.3:
            if primary_emotion in ['anger', 'frustration']:
                return "empathetic"
            else:
                return "apologetic"
        
        # Negative sentiment
        elif sentiment_score < 0.5:
            return "empathetic"
        
        # Neutral to positive sentiment
        elif sentiment_score < 0.7:
            return "professional"
        
        # Very positive sentiment
        else:
            return "reassuring"
    
    def _prepare_context(self, retrieved_docs: List[Dict]) -> str:
        """Prepare context from retrieved documents"""
        if not retrieved_docs:
            return "No specific knowledge base articles found for this query."
        
        context_parts = []
        for i, doc in enumerate(retrieved_docs[:3]):  # Use top 3 most relevant
            context_parts.append(f"Article {i+1}: {doc['title']}\n{doc['content'][:500]}...")
        
        return "\n\n".join(context_parts)
    
    def _generate_ai_response(self, customer_message: str, context: str, 
                             conversation_history: List[Dict], tone: str, sentiment_data: Dict) -> str:
        """Generate AI response with appropriate tone and context"""
        try:
            # Prepare conversation context
            recent_context = self._format_conversation_context(conversation_history[-4:])
            
            # Get tone description
            tone_description = self.tone_styles.get(tone, "professional and helpful")
            
            # Build comprehensive prompt
            prompt = f"""
            You are a customer support AI assistant. Generate a helpful, {tone_description} response to the customer.
            
            Customer's current message: "{customer_message}"
            
            Customer sentiment analysis:
            - Sentiment score: {sentiment_data.get('sentiment_score', 0.5):.2f} (0=very negative, 1=very positive)
            - Primary emotion: {sentiment_data.get('primary_emotion', 'neutral')}
            - Urgency level: {sentiment_data.get('urgency', 0.5):.2f}
            
            Recent conversation context:
            {recent_context}
            
            Relevant knowledge base information:
            {context}
            
            Guidelines for your response:
            1. Be {tone_description}
            2. Address the customer's specific concern directly
            3. Use information from the knowledge base when relevant
            4. Show empathy appropriate to their emotional state
            5. Provide clear, actionable next steps
            6. Keep response concise but complete (aim for 2-3 sentences)
            7. If escalation is needed, acknowledge their frustration and offer escalation
            
            Generate a response that feels natural and human-like:
            """
            
            response = self.openai_client.chat.completions.create(
                model=config.CHAT_MODEL,  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert customer support representative. "
                        "Provide helpful, empathetic, and professional responses. "
                        "Always acknowledge the customer's feelings and provide clear solutions."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=config.TEMPERATURE,
                max_tokens=config.MAX_RESPONSE_LENGTH
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            st.error(f"AI response generation failed: {str(e)}")
            return self._generate_fallback_response(customer_message, tone)
    
    def _format_conversation_context(self, recent_messages: List[Dict]) -> str:
        """Format recent conversation for context"""
        if not recent_messages:
            return "This is the start of the conversation."
        
        formatted = []
        for msg in recent_messages:
            sender = "Customer" if msg['sender'] == 'customer' else "Agent"
            formatted.append(f"{sender}: {msg['content']}")
        
        return "\n".join(formatted)
    
    def _generate_fallback_response(self, customer_message: str, tone: str) -> str:
        """Generate fallback response when AI fails"""
        fallback_responses = {
            "empathetic": "I understand your concern and I'm here to help. Let me look into this for you right away.",
            "professional": "Thank you for contacting us. I'll be happy to assist you with your inquiry.",
            "apologetic": "I apologize for any inconvenience this has caused. Let me work on resolving this issue for you.",
            "reassuring": "I'm confident we can resolve this together. Let me guide you through the next steps.",
            "urgent": "I understand this is urgent for you. Let me prioritize your request and get this resolved quickly."
        }
        
        return fallback_responses.get(tone, "Thank you for your message. I'm here to help you with your inquiry.")
    
    def calibrate_response_tone(self, base_response: str, target_sentiment: float, 
                               current_tone: str) -> str:
        """Calibrate response tone based on desired sentiment outcome"""
        try:
            if abs(target_sentiment - 0.7) < 0.1:  # Target is neutral-positive
                return base_response
            
            prompt = f"""
            Adjust the tone of this customer support response to be more appropriate for achieving a target sentiment of {target_sentiment:.2f} (0=negative, 1=positive).
            
            Original response: "{base_response}"
            Current tone: {current_tone}
            
            Guidelines:
            - If target sentiment is low (< 0.4), be more apologetic and empathetic
            - If target sentiment is high (> 0.7), be more positive and reassuring
            - Maintain the same factual content and helpfulness
            - Keep the response length similar
            
            Provide the adjusted response:
            """
            
            response = self.openai_client.chat.completions.create(
                model=config.CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=config.MAX_RESPONSE_LENGTH
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            st.error(f"Tone calibration failed: {str(e)}")
            return base_response
    
    def generate_escalation_response(self, escalation_level: str, customer_message: str) -> str:
        """Generate specific response for escalation scenarios"""
        try:
            escalation_prompts = {
                "medium": "The customer seems moderately frustrated. Acknowledge their concern and offer additional assistance.",
                "high": "The customer is quite frustrated. Show empathy, apologize if appropriate, and consider escalation options.",
                "critical": "The customer is very upset and likely to escalate. Provide immediate empathy, sincere apology, and escalation path."
            }
            
            prompt = f"""
            Generate an escalation-appropriate response for this customer support scenario.
            
            Escalation level: {escalation_level}
            Customer message: "{customer_message}"
            
            {escalation_prompts.get(escalation_level, "Provide a professional and helpful response.")}
            
            The response should:
            1. Acknowledge their frustration appropriately
            2. Take responsibility where applicable
            3. Offer concrete next steps
            4. Show genuine care for their experience
            
            Generate a response:
            """
            
            response = self.openai_client.chat.completions.create(
                model=config.CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                max_tokens=config.MAX_RESPONSE_LENGTH
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            st.error(f"Escalation response generation failed: {str(e)}")
            return self.escalation_responses.get(escalation_level, 
                "I understand your concern and want to make sure we address this properly for you.")
    
    def is_healthy(self) -> bool:
        """Check if response generator is working properly"""
        try:
            # Test with simple generation
            test_response = self._generate_ai_response(
                "Hello", "Test context", [], "professional", {"sentiment_score": 0.5}
            )
            return len(test_response) > 0
        except:
            return False
