"""
Intent Classification + Domain Detection
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class IntentClassifier:
    """Classifies user intent + Auto-detects database domain"""
    
    def __init__(self):
        # Load sentence transformer model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Intent definitions
        self.intents = {
            "generate_report": ["generate report", "create report", "show report", "make report"],
            "query_data": ["show me", "get data", "what is", "display", "fetch", "retrieve"],
            "analyze_metrics": ["total", "sum", "average", "count", "calculate", "compute"],
            "compare": ["compare", "versus", "vs", "difference", "between"],
            "trend": ["trend", "over time", "growth", "change", "timeline"],
            "filter": ["filter", "where", "only", "specific", "particular"],
            "top_bottom": ["top", "bottom", "best", "worst", "highest", "lowest"]
        }
        
        # Domain definitions (expandable!)
        self.domain_signatures = {
            'healthcare': {
                'keywords': [
                    'patient', 'diagnosis', 'treatment', 'medication', 'prescription',
                    'doctor', 'physician', 'medical', 'clinic', 'symptom', 'disease',
                    'icd', 'cpt', 'vital_signs', 'blood_pressure', 'heart_rate'
                ],
                'description': 'medical healthcare patient diagnosis treatment doctor hospital pharmacy medicine clinical'
            },
            'finance': {
                'keywords': [
                    'transaction', 'account', 'balance', 'revenue', 'profit', 'loss',
                    'invoice', 'payment', 'credit', 'debit', 'ledger', 'expense',
                    'income', 'budget', 'investment', 'loan', 'interest'
                ],
                'description': 'financial money transaction account balance payment invoice revenue profit banking investment'
            },
            'hospital': {
                'keywords': [
                    'admission', 'discharge', 'ward', 'bed', 'nurse', 'emergency',
                    'surgery', 'radiology', 'lab_test', 'icu', 'operation',
                    'inpatient', 'outpatient', 'appointment', 'room'
                ],
                'description': 'hospital admission patient ward bed nurse emergency surgery medical records department'
            },
            'retail': {
                'keywords': [
                    'product', 'sales', 'customer', 'order', 'inventory', 'purchase',
                    'sku', 'price', 'quantity', 'store', 'cart', 'checkout',
                    'shipping', 'delivery', 'supplier', 'stock'
                ],
                'description': 'retail sales product customer order inventory purchase shopping ecommerce store'
            },
            'education': {
                'keywords': [
                    'student', 'teacher', 'course', 'grade', 'exam', 'assignment',
                    'enrollment', 'class', 'subject', 'semester', 'department',
                    'faculty', 'attendance', 'marks', 'gpa'
                ],
                'description': 'education student teacher course grade exam school university college learning'
            },
            'hr': {
                'keywords': [
                    'employee', 'salary', 'department', 'payroll', 'leave', 'attendance',
                    'recruitment', 'performance', 'appraisal', 'manager', 'hire',
                    'resignation', 'promotion', 'bonus', 'benefits'
                ],
                'description': 'human resources employee salary department payroll recruitment performance management'
            },
            'logistics': {
                'keywords': [
                    'shipment', 'delivery', 'warehouse', 'transport', 'tracking',
                    'carrier', 'freight', 'route', 'vehicle', 'driver', 'cargo',
                    'dispatch', 'loading', 'unloading', 'inventory'
                ],
                'description': 'logistics shipping delivery warehouse transport tracking freight supply chain'
            },
            'ecommerce': {
                'keywords': [
                    'cart', 'checkout', 'payment', 'order', 'customer', 'product',
                    'review', 'rating', 'wishlist', 'discount', 'coupon', 'refund',
                    'shipping', 'returns', 'browse'
                ],
                'description': 'ecommerce online shopping cart checkout payment order customer product website'
            }
        }
    
    def classify(self, prompt: str) -> Dict:
        """Classify intent from user prompt"""
        
        prompt_lower = prompt.lower()
        prompt_emb = self.model.encode(prompt_lower)
        
        # Calculate similarities
        scores = {}
        for intent, keywords in self.intents.items():
            intent_text = " ".join(keywords)
            intent_emb = self.model.encode(intent_text)
            
            # Cosine similarity
            similarity = np.dot(prompt_emb, intent_emb) / (
                np.linalg.norm(prompt_emb) * np.linalg.norm(intent_emb)
            )
            scores[intent] = float(similarity)
        
        best_intent = max(scores, key=scores.get)
        
        return {
            "intent": best_intent,
            "confidence": scores[best_intent],
            "all_scores": scores
        }
    
    def detect_domain(self, schema: Dict) -> Tuple[str, float, Dict]:
        """
        🔥 AUTO-DETECT DATABASE DOMAIN
        
        Args:
            schema: Database schema dict
            
        Returns:
            (domain_name, confidence_score, all_scores)
        """
        
        # Extract all text from schema
        schema_text = self._schema_to_text(schema)
        
        # Method 1: Keyword matching (fast, reliable)
        keyword_scores = self._keyword_based_detection(schema_text)
        
        # Method 2: Semantic similarity (AI-based, contextual)
        semantic_scores = self._semantic_based_detection(schema_text)
        
        # Combine scores (70% semantic + 30% keyword)
        combined_scores = {}
        for domain in self.domain_signatures.keys():
            combined_scores[domain] = (
                0.7 * semantic_scores.get(domain, 0) +
                0.3 * keyword_scores.get(domain, 0)
            )
        
        # Add 'general' domain if no strong match
        max_score = max(combined_scores.values()) if combined_scores else 0
        if max_score < 0.3:
            return 'general', max_score, combined_scores
        
        best_domain = max(combined_scores, key=combined_scores.get)
        
        return best_domain, combined_scores[best_domain], combined_scores
    
    def _schema_to_text(self, schema: Dict) -> str:
        """Convert schema to text for analysis"""
        text_parts = []
        for table, columns in schema.items():
            text_parts.append(table)
            text_parts.extend(columns)
        return " ".join(text_parts).lower()
    
    def _keyword_based_detection(self, schema_text: str) -> Dict[str, float]:
        """Keyword-based domain detection"""
        scores = {}
        
        for domain, config in self.domain_signatures.items():
            matches = sum(1 for kw in config['keywords'] if kw in schema_text)
            total_keywords = len(config['keywords'])
            scores[domain] = matches / total_keywords if total_keywords > 0 else 0
        
        return scores
    
    def _semantic_based_detection(self, schema_text: str) -> Dict[str, float]:
        """AI-based semantic domain detection"""
        schema_emb = self.model.encode(schema_text)
        scores = {}
        
        for domain, config in self.domain_signatures.items():
            domain_emb = self.model.encode(config['description'])
            
            # Cosine similarity
            similarity = np.dot(schema_emb, domain_emb) / (
                np.linalg.norm(schema_emb) * np.linalg.norm(domain_emb)
            )
            scores[domain] = float(similarity)
        
        return scores
    
    def extract_entities(self, prompt: str) -> Dict:
        """Extract entities (metrics, dimensions, time period, etc.)"""
        
        entities = {
            "metrics": [],
            "dimensions": [],
            "time_period": None,
            "aggregation": None,
            "limit": None
        }
        
        prompt_lower = prompt.lower()
        
        # Extract metrics (domain-agnostic)
        metric_patterns = [
            'total', 'sum', 'count', 'average', 'max', 'min',
            'revenue', 'sales', 'profit', 'cost', 'price', 'amount',
            'quantity', 'number', 'rate', 'percentage'
        ]
        entities["metrics"] = [m for m in metric_patterns if m in prompt_lower]
        
        # Extract dimensions
        dimension_patterns = [
            'product', 'customer', 'region', 'category', 'type',
            'date', 'time', 'month', 'year', 'day', 'department',
            'location', 'city', 'country', 'state'
        ]
        entities["dimensions"] = [d for d in dimension_patterns if d in prompt_lower]
        
        # Extract time period
        time_map = {
            "last_month": ["last month", "previous month"],
            "current_month": ["this month", "current month"],
            "last_year": ["last year", "previous year"],
            "current_year": ["this year", "current year"],
            "last_quarter": ["last quarter", "previous quarter"],
            "last_week": ["last week", "previous week"]
        }
        
        for period, keywords in time_map.items():
            if any(kw in prompt_lower for kw in keywords):
                entities["time_period"] = period
                break
        
        # Extract aggregation
        if any(word in prompt_lower for word in ["total", "sum"]):
            entities["aggregation"] = "sum"
        elif any(word in prompt_lower for word in ["average", "avg", "mean"]):
            entities["aggregation"] = "average"
        elif "count" in prompt_lower:
            entities["aggregation"] = "count"
        elif any(word in prompt_lower for word in ["max", "maximum", "highest"]):
            entities["aggregation"] = "max"
        elif any(word in prompt_lower for word in ["min", "minimum", "lowest"]):
            entities["aggregation"] = "min"
        
        # Extract limit (top N, bottom N)
        import re
        numbers = re.findall(r'\b(\d+)\b', prompt_lower)
        if numbers and any(word in prompt_lower for word in ['top', 'bottom', 'first', 'last']):
            entities["limit"] = int(numbers[0])
        
        return entities