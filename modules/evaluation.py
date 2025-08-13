import json
import time
import numpy as np
from typing import List, Dict, Any, Optional
from openai import OpenAI
import streamlit as st
import config

class RAGEvaluator:
    def __init__(self):
        """Initialize RAGAS evaluator"""
        self.openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
        
        # Evaluation metrics
        self.metrics = {
            "context_precision": 0.0,
            "context_recall": 0.0,
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "retrieval_accuracy": 0.0,
            "response_latency": 0.0
        }
        
        # Evaluation history
        self.evaluation_history = []
    
    def evaluate_response(self, query: str, response: str, retrieved_docs: List[Dict], 
                         ground_truth: Optional[str] = None) -> Dict[str, Any]:
        """Comprehensive evaluation using RAGAS-inspired metrics"""
        try:
            start_time = time.time()
            
            # Evaluate different aspects
            context_precision = self._evaluate_context_precision(query, retrieved_docs)
            context_recall = self._evaluate_context_recall(query, retrieved_docs, ground_truth)
            faithfulness = self._evaluate_faithfulness(response, retrieved_docs)
            answer_relevancy = self._evaluate_answer_relevancy(query, response)
            retrieval_accuracy = self._evaluate_retrieval_accuracy(query, retrieved_docs)
            
            evaluation_time = time.time() - start_time
            
            # Compile results
            evaluation_result = {
                "timestamp": time.time(),
                "query": query,
                "response": response,
                "metrics": {
                    "context_precision": context_precision,
                    "context_recall": context_recall,
                    "faithfulness": faithfulness,
                    "answer_relevancy": answer_relevancy,
                    "retrieval_accuracy": retrieval_accuracy,
                    "evaluation_latency": evaluation_time
                },
                "retrieved_docs_count": len(retrieved_docs),
                "overall_score": self._calculate_overall_score({
                    "context_precision": context_precision,
                    "faithfulness": faithfulness,
                    "answer_relevancy": answer_relevancy,
                    "retrieval_accuracy": retrieval_accuracy
                })
            }
            
            # Store in history
            self.evaluation_history.append(evaluation_result)
            
            return evaluation_result
            
        except Exception as e:
            st.error(f"Evaluation failed: {str(e)}")
            return {
                "timestamp": time.time(),
                "query": query,
                "response": response,
                "metrics": {
                    "context_precision": 0.5,
                    "context_recall": 0.5,
                    "faithfulness": 0.5,
                    "answer_relevancy": 0.5,
                    "retrieval_accuracy": 0.5,
                    "evaluation_latency": 0.0
                },
                "overall_score": 0.5,
                "error": str(e)
            }
    
    def _evaluate_context_precision(self, query: str, retrieved_docs: List[Dict]) -> float:
        """Evaluate how precise the retrieved context is"""
        if not retrieved_docs:
            return 0.0
        
        try:
            # Create context from retrieved documents
            context = "\n".join([doc.get('content', '') for doc in retrieved_docs])
            
            prompt = f"""
            Evaluate the precision of the retrieved context for answering the query.
            
            Query: "{query}"
            
            Retrieved Context:
            {context[:2000]}...
            
            Rate the precision from 0.0 to 1.0 based on:
            1. How relevant each piece of context is to the query
            2. How much irrelevant information is included
            3. Whether the context directly addresses the query
            
            Respond with JSON:
            {{
                "precision_score": number,
                "reasoning": "explanation"
            }}
            """
            
            response = self.openai_client.chat.completions.create(
                model=config.CHAT_MODEL,  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.2
            )
            
            result = json.loads(response.choices[0].message.content)
            return max(0.0, min(1.0, result.get("precision_score", 0.5)))
            
        except Exception as e:
            st.warning(f"Context precision evaluation failed: {str(e)}")
            return 0.5
    
    def _evaluate_context_recall(self, query: str, retrieved_docs: List[Dict], 
                                ground_truth: Optional[str] = None) -> float:
        """Evaluate how much relevant context was retrieved"""
        if not retrieved_docs:
            return 0.0
        
        try:
            # If no ground truth, use heuristic evaluation
            if not ground_truth:
                # Simple heuristic: more relevant docs = better recall
                relevance_scores = [doc.get('score', 0.5) for doc in retrieved_docs]
                avg_relevance = sum(relevance_scores) / len(relevance_scores)
                return min(1.0, avg_relevance * 1.2)  # Scale up slightly
            
            # With ground truth, compare coverage
            context = "\n".join([doc.get('content', '') for doc in retrieved_docs])
            
            prompt = f"""
            Evaluate how well the retrieved context covers the information needed to answer the query.
            
            Query: "{query}"
            Ground Truth Answer: "{ground_truth}"
            
            Retrieved Context:
            {context[:2000]}...
            
            Rate the recall from 0.0 to 1.0 based on:
            1. How much of the needed information is present in the context
            2. Whether key facts from ground truth are covered
            3. Completeness of the retrieved information
            
            Respond with JSON:
            {{
                "recall_score": number,
                "reasoning": "explanation"
            }}
            """
            
            response = self.openai_client.chat.completions.create(
                model=config.CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.2
            )
            
            result = json.loads(response.choices[0].message.content)
            return max(0.0, min(1.0, result.get("recall_score", 0.5)))
            
        except Exception as e:
            st.warning(f"Context recall evaluation failed: {str(e)}")
            return 0.5
    
    def _evaluate_faithfulness(self, response: str, retrieved_docs: List[Dict]) -> float:
        """Evaluate how faithful the response is to the retrieved context"""
        if not retrieved_docs:
            return 0.5
        
        try:
            context = "\n".join([doc.get('content', '') for doc in retrieved_docs])
            
            prompt = f"""
            Evaluate how faithful the response is to the provided context.
            
            Context:
            {context[:2000]}...
            
            Response: "{response}"
            
            Rate the faithfulness from 0.0 to 1.0 based on:
            1. Whether the response contains information not in the context (hallucination)
            2. How accurately the response reflects the context
            3. Whether facts are correctly represented
            
            Respond with JSON:
            {{
                "faithfulness_score": number,
                "reasoning": "explanation"
            }}
            """
            
            response_eval = self.openai_client.chat.completions.create(
                model=config.CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.2
            )
            
            result = json.loads(response_eval.choices[0].message.content)
            return max(0.0, min(1.0, result.get("faithfulness_score", 0.5)))
            
        except Exception as e:
            st.warning(f"Faithfulness evaluation failed: {str(e)}")
            return 0.5
    
    def _evaluate_answer_relevancy(self, query: str, response: str) -> float:
        """Evaluate how relevant the response is to the query"""
        try:
            prompt = f"""
            Evaluate how relevant and helpful the response is for the given query.
            
            Query: "{query}"
            Response: "{response}"
            
            Rate the relevancy from 0.0 to 1.0 based on:
            1. How directly the response addresses the query
            2. Whether the response provides useful information
            3. How well the response would satisfy the user's need
            
            Respond with JSON:
            {{
                "relevancy_score": number,
                "reasoning": "explanation"
            }}
            """
            
            response_eval = self.openai_client.chat.completions.create(
                model=config.CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.2
            )
            
            result = json.loads(response_eval.choices[0].message.content)
            return max(0.0, min(1.0, result.get("relevancy_score", 0.5)))
            
        except Exception as e:
            st.warning(f"Answer relevancy evaluation failed: {str(e)}")
            return 0.5
    
    def _evaluate_retrieval_accuracy(self, query: str, retrieved_docs: List[Dict]) -> float:
        """Evaluate accuracy of document retrieval"""
        if not retrieved_docs:
            return 0.0
        
        try:
            # Evaluate based on relevance scores and content quality
            relevance_scores = []
            
            for doc in retrieved_docs:
                doc_score = doc.get('score', 0.5)
                
                # Additional content relevance check
                prompt = f"""
                Rate how relevant this document is to the query on a scale of 0.0 to 1.0.
                
                Query: "{query}"
                Document Title: "{doc.get('title', '')}"
                Document Content: "{doc.get('content', '')[:500]}..."
                
                Respond with JSON:
                {{
                    "relevance_score": number
                }}
                """
                
                try:
                    response = self.openai_client.chat.completions.create(
                        model=config.CHAT_MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        response_format={"type": "json_object"},
                        temperature=0.2
                    )
                    
                    ai_result = json.loads(response.choices[0].message.content)
                    ai_score = ai_result.get("relevance_score", 0.5)
                    
                    # Combine vector similarity score with AI evaluation
                    combined_score = (doc_score + ai_score) / 2
                    relevance_scores.append(combined_score)
                    
                except:
                    relevance_scores.append(doc_score)
            
            # Calculate weighted average (higher weight for top results)
            weights = [1.0 / (i + 1) for i in range(len(relevance_scores))]
            weighted_score = sum(score * weight for score, weight in zip(relevance_scores, weights))
            total_weight = sum(weights)
            
            return weighted_score / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            st.warning(f"Retrieval accuracy evaluation failed: {str(e)}")
            return 0.5
    
    def _calculate_overall_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall score from individual metrics"""
        # Weighted average of metrics
        weights = {
            "context_precision": 0.2,
            "faithfulness": 0.3,
            "answer_relevancy": 0.3,
            "retrieval_accuracy": 0.2
        }
        
        weighted_sum = sum(metrics.get(metric, 0.0) * weight for metric, weight in weights.items())
        return weighted_sum
    
    def get_evaluation_summary(self, last_n: int = 10) -> Dict[str, Any]:
        """Get summary of recent evaluations"""
        if not self.evaluation_history:
            return {"message": "No evaluations available"}
        
        recent_evaluations = self.evaluation_history[-last_n:]
        
        # Calculate averages
        avg_metrics = {}
        for metric in ["context_precision", "faithfulness", "answer_relevancy", "retrieval_accuracy"]:
            scores = [eval_result["metrics"].get(metric, 0.0) for eval_result in recent_evaluations]
            avg_metrics[metric] = sum(scores) / len(scores) if scores else 0.0
        
        # Calculate trends
        if len(recent_evaluations) >= 5:
            recent_half = recent_evaluations[len(recent_evaluations)//2:]
            earlier_half = recent_evaluations[:len(recent_evaluations)//2]
            
            recent_avg = sum(eval_result["overall_score"] for eval_result in recent_half) / len(recent_half)
            earlier_avg = sum(eval_result["overall_score"] for eval_result in earlier_half) / len(earlier_half)
            
            trend = "improving" if recent_avg > earlier_avg + 0.05 else "declining" if recent_avg < earlier_avg - 0.05 else "stable"
        else:
            trend = "insufficient_data"
        
        return {
            "evaluation_count": len(recent_evaluations),
            "average_metrics": avg_metrics,
            "overall_average": sum(eval_result["overall_score"] for eval_result in recent_evaluations) / len(recent_evaluations),
            "trend": trend,
            "last_evaluation": recent_evaluations[-1]["timestamp"] if recent_evaluations else None
        }
    
    def export_evaluation_data(self) -> str:
        """Export evaluation data as JSON"""
        return json.dumps(self.evaluation_history, indent=2)
