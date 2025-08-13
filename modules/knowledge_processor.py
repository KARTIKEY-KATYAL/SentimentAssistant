import json
import os
import time
from typing import List, Dict, Any
import streamlit as st

class KnowledgeProcessor:
    def __init__(self):
        """Initialize knowledge processor"""
        self.knowledge_base_path = "data/sample_knowledge_base.json"
        self.processed_docs = []
    
    def load_knowledge_base(self) -> bool:
        """Load and process knowledge base documents"""
        try:
            # Load from file if exists
            if os.path.exists(self.knowledge_base_path):
                with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
                    knowledge_data = json.load(f)
            else:
                # Create sample knowledge base if file doesn't exist
                knowledge_data = self._create_sample_knowledge_base()
                self._save_knowledge_base(knowledge_data)
            
            # Process documents for vector storage
            self.processed_docs = self._process_documents(knowledge_data.get('articles', []))
            
            # Store in vector database
            from modules.vector_store import VectorStore
            vector_store = VectorStore()
            success = vector_store.upsert_documents(self.processed_docs)
            
            if success:
                st.success(f"Loaded {len(self.processed_docs)} knowledge base articles")
                return True
            else:
                st.error("Failed to store documents in vector database")
                return False
                
        except Exception as e:
            st.error(f"Failed to load knowledge base: {str(e)}")
            return False
    
    def _create_sample_knowledge_base(self) -> Dict:
        """Create sample knowledge base for demonstration"""
        return {
            "articles": [
                {
                    "id": "kb_001",
                    "title": "How to Reset Your Password",
                    "content": "To reset your password, follow these steps: 1. Go to the login page 2. Click 'Forgot Password' 3. Enter your email address 4. Check your email for reset instructions 5. Follow the link in the email 6. Create a new strong password. If you don't receive the email within 10 minutes, check your spam folder or contact support.",
                    "category": "account_management",
                    "tags": ["password", "reset", "login", "account"],
                    "priority": "high",
                    "last_updated": "2024-01-15"
                },
                {
                    "id": "kb_002", 
                    "title": "Billing and Payment Issues",
                    "content": "Common billing issues and solutions: Payment failed - Check if your card details are correct and up to date. Unexpected charges - Review your subscription plan and any add-ons. Refund requests - Contact support within 30 days with your order number. Payment methods - We accept major credit cards, PayPal, and bank transfers. For enterprise customers, we also offer invoicing options.",
                    "category": "billing",
                    "tags": ["billing", "payment", "refund", "subscription", "charges"],
                    "priority": "high",
                    "last_updated": "2024-01-20"
                },
                {
                    "id": "kb_003",
                    "title": "Account Suspension and Recovery", 
                    "content": "If your account has been suspended: 1. Check your email for suspension notification 2. Common reasons include policy violations, payment issues, or security concerns 3. Contact support immediately with your account details 4. Provide any requested documentation 5. Follow the recovery process outlined in our email 6. Account recovery typically takes 2-5 business days. Prevent future suspensions by keeping your account information current and following our terms of service.",
                    "category": "account_management",
                    "tags": ["suspension", "account", "recovery", "policy", "security"],
                    "priority": "critical",
                    "last_updated": "2024-01-18"
                },
                {
                    "id": "kb_004",
                    "title": "Technical Support and Troubleshooting",
                    "content": "Basic troubleshooting steps: 1. Clear your browser cache and cookies 2. Try using an incognito/private browser window 3. Disable browser extensions temporarily 4. Check your internet connection 5. Try accessing from a different device or browser 6. Update your browser to the latest version. If problems persist, contact technical support with details about your device, browser, and the specific error you're experiencing.",
                    "category": "technical_support",
                    "tags": ["troubleshooting", "technical", "browser", "cache", "error"],
                    "priority": "medium",
                    "last_updated": "2024-01-10"
                },
                {
                    "id": "kb_005",
                    "title": "Privacy and Data Protection",
                    "content": "We take your privacy seriously. Your data protection rights include: Access - Request copies of your personal data. Correction - Request correction of inaccurate data. Deletion - Request deletion of your data (right to be forgotten). Portability - Request transfer of your data. We use industry-standard encryption and security measures. Data is stored securely and never sold to third parties. For privacy concerns or data requests, contact our privacy team at privacy@company.com.",
                    "category": "privacy",
                    "tags": ["privacy", "data", "security", "GDPR", "rights"],
                    "priority": "medium",
                    "last_updated": "2024-01-12"
                },
                {
                    "id": "kb_006",
                    "title": "Subscription Management and Plans",
                    "content": "Manage your subscription: Upgrade/Downgrade - Changes take effect at next billing cycle. Cancel subscription - You can cancel anytime; access continues until period end. Plan comparison - Basic ($9/month), Pro ($19/month), Enterprise (custom pricing). Features include: Basic (core features), Pro (advanced features + priority support), Enterprise (all features + dedicated support + custom integrations). Contact sales for enterprise pricing and custom solutions.",
                    "category": "subscription",
                    "tags": ["subscription", "plans", "upgrade", "cancel", "pricing"],
                    "priority": "high",
                    "last_updated": "2024-01-25"
                },
                {
                    "id": "kb_007",
                    "title": "API Documentation and Integration",
                    "content": "API integration guide: 1. Obtain API key from your dashboard 2. Authentication uses Bearer token in headers 3. Base URL: https://api.example.com/v1 4. Rate limits: 1000 requests/hour for Basic, 5000/hour for Pro 5. Key endpoints: /users, /data, /analytics 6. Response format is JSON with standard HTTP status codes 7. SDKs available for Python, JavaScript, and PHP. For technical integration support, contact our developer support team.",
                    "category": "technical_support",
                    "tags": ["API", "integration", "developer", "authentication", "documentation"],
                    "priority": "medium",
                    "last_updated": "2024-01-22"
                },
                {
                    "id": "kb_008",
                    "title": "Refund and Cancellation Policy",
                    "content": "Refund policy: 30-day money-back guarantee for new subscriptions. Cancellation policy: Cancel anytime, no cancellation fees. Refund process: 1. Contact support within 30 days 2. Provide order number and reason 3. Refunds processed within 5-7 business days 4. Refunds issued to original payment method. Exceptions: Setup fees for enterprise plans are non-refundable. Partial refunds available for annual plans if cancelled within first 30 days.",
                    "category": "billing",
                    "tags": ["refund", "cancellation", "policy", "money-back", "guarantee"],
                    "priority": "high",
                    "last_updated": "2024-01-20"
                },
                {
                    "id": "kb_009",
                    "title": "Security Best Practices",
                    "content": "Keep your account secure: 1. Use strong, unique passwords 2. Enable two-factor authentication (2FA) 3. Regularly review account activity 4. Don't share login credentials 5. Log out from shared devices 6. Report suspicious activity immediately 7. Keep contact information updated 8. Use official company communications only. If you suspect account compromise, change your password immediately and contact security team at security@company.com.",
                    "category": "security",
                    "tags": ["security", "password", "2FA", "best-practices", "account-safety"],
                    "priority": "high",
                    "last_updated": "2024-01-15"
                },
                {
                    "id": "kb_010",
                    "title": "Contact Information and Support Hours",
                    "content": "Get support when you need it: Email support: support@company.com (24/7). Live chat: Available Mon-Fri 9AM-6PM EST. Phone support: +1-800-123-4567 (Pro and Enterprise only). Emergency support: For critical issues, mark emails as 'URGENT'. Response times: Basic (48 hours), Pro (24 hours), Enterprise (4 hours). Self-service: Check our knowledge base first for quick answers. Status page: status.company.com for service updates.",
                    "category": "support",
                    "tags": ["contact", "support", "hours", "email", "phone", "chat"],
                    "priority": "critical",
                    "last_updated": "2024-01-28"
                }
            ]
        }
    
    def _process_documents(self, articles: List[Dict]) -> List[Dict]:
        """Process articles for vector storage"""
        processed = []
        
        for article in articles:
            # Create document for vector storage
            doc = {
                "id": article.get("id", f"doc_{int(time.time())}"),
                "title": article.get("title", ""),
                "content": article.get("content", ""),
                "category": article.get("category", "general"),
                "tags": article.get("tags", []),
                "priority": article.get("priority", "medium"),
                "created_at": time.time(),
                "last_updated": article.get("last_updated", "")
            }
            
            # Create additional searchable content by combining title and content
            searchable_content = f"{doc['title']} {doc['content']}"
            doc["searchable_content"] = searchable_content
            
            # Use content for embedding generation
            doc["content"] = searchable_content
            
            processed.append(doc)
        
        return processed
    
    def _save_knowledge_base(self, data: Dict):
        """Save knowledge base to file"""
        try:
            os.makedirs(os.path.dirname(self.knowledge_base_path), exist_ok=True)
            with open(self.knowledge_base_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            st.error(f"Failed to save knowledge base: {str(e)}")
    
    def add_article(self, article: Dict) -> bool:
        """Add new article to knowledge base"""
        try:
            # Load existing data
            if os.path.exists(self.knowledge_base_path):
                with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                data = {"articles": []}
            
            # Add new article
            article["id"] = f"kb_{int(time.time())}"
            article["last_updated"] = time.strftime("%Y-%m-%d")
            data["articles"].append(article)
            
            # Save updated data
            self._save_knowledge_base(data)
            
            # Process and store in vector database
            processed_doc = self._process_documents([article])[0]
            from modules.vector_store import VectorStore
            vector_store = VectorStore()
            return vector_store.upsert_documents([processed_doc])
            
        except Exception as e:
            st.error(f"Failed to add article: {str(e)}")
            return False
    
    def update_article(self, article_id: str, updated_data: Dict) -> bool:
        """Update existing article"""
        try:
            if not os.path.exists(self.knowledge_base_path):
                return False
            
            with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Find and update article
            for i, article in enumerate(data["articles"]):
                if article["id"] == article_id:
                    data["articles"][i].update(updated_data)
                    data["articles"][i]["last_updated"] = time.strftime("%Y-%m-%d")
                    break
            else:
                return False
            
            # Save updated data
            self._save_knowledge_base(data)
            
            # Update in vector database
            updated_article = next(a for a in data["articles"] if a["id"] == article_id)
            processed_doc = self._process_documents([updated_article])[0]
            from modules.vector_store import VectorStore
            vector_store = VectorStore()
            return vector_store.upsert_documents([processed_doc])
            
        except Exception as e:
            st.error(f"Failed to update article: {str(e)}")
            return False
    
    def delete_article(self, article_id: str) -> bool:
        """Delete article from knowledge base"""
        try:
            if not os.path.exists(self.knowledge_base_path):
                return False
            
            with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Remove article
            data["articles"] = [a for a in data["articles"] if a["id"] != article_id]
            
            # Save updated data
            self._save_knowledge_base(data)
            
            # Delete from vector database
            from modules.vector_store import VectorStore
            vector_store = VectorStore()
            return vector_store.delete_document(article_id)
            
        except Exception as e:
            st.error(f"Failed to delete article: {str(e)}")
            return False
    
    def get_article_stats(self) -> Dict:
        """Get knowledge base statistics"""
        try:
            if not os.path.exists(self.knowledge_base_path):
                return {}
            
            with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            articles = data.get("articles", [])
            
            # Calculate statistics
            stats = {
                "total_articles": len(articles),
                "categories": {},
                "priorities": {},
                "recent_updates": 0
            }
            
            current_date = time.strftime("%Y-%m-%d")
            
            for article in articles:
                # Category distribution
                category = article.get("category", "unknown")
                stats["categories"][category] = stats["categories"].get(category, 0) + 1
                
                # Priority distribution
                priority = article.get("priority", "medium")
                stats["priorities"][priority] = stats["priorities"].get(priority, 0) + 1
                
                # Recent updates (within last 7 days)
                last_updated = article.get("last_updated", "")
                if last_updated == current_date:
                    stats["recent_updates"] += 1
            
            return stats
            
        except Exception as e:
            st.error(f"Failed to get article stats: {str(e)}")
            return {}
