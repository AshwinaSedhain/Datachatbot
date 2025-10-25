"""
Main Chatbot Agent - Domain-Aware Orchestration
"""

from typing import Dict, Any, Callable, Optional
from .intent_classifier import IntentClassifier
from .query_generator import QueryGenerator
from .response_generator import ResponseGenerator
from .visualizer import AutoVisualizer


class ChatbotAgent:
    """
    Universal AI Chatbot Agent
    
    ✅ Works with ANY database (auto-detects domain)
    ✅ Adapts responses based on domain
    """
    
    def __init__(self):
        print("\n" + "="*70)
        print("🤖 Initializing Universal AI Chatbot Agent...")
        print("="*70)
        
        print("Loading components...")
        self.intent_classifier = IntentClassifier()
        print("✅ Intent Classifier ready (with domain detection)")
        
        self.query_generator = QueryGenerator()
        print("✅ Query Generator ready")
        
        self.response_generator = ResponseGenerator()
        print("✅ Response Generator ready")

        self.visualizer = AutoVisualizer()
        print("✅ Visualizer ready")
        
        print("="*70)
        print("✅ Universal Chatbot Agent initialized!")
        print("   Supports: Healthcare, Finance, Retail, Education, HR, Logistics, E-commerce, and more!")
        print("="*70 + "\n")
    
    def process(
        self,
        user_prompt: str,
        database_schema: Dict,
        execute_query: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Process user prompt with automatic domain detection
        
        Args:
            user_prompt (str): Natural language question from user
                Example: "Show me total sales by product"
            
            database_schema (dict): Database schema mapping
                Example: {
                    "sales": ["id", "date", "product", "amount", "region"],
                    "customers": ["id", "name", "email", "region"]
                }
            
            execute_query (callable, optional): Function to execute SQL
                Example: lambda sql: pandas.read_sql(sql, db_connection)
        
        Returns:
            dict: {
                "success": bool,
                "domain": str,                  # 🔥 Auto-detected domain
                "domain_confidence": float,     # 🔥 Detection confidence
                "intent": dict,
                "generated_query": str,
                "query_results": Any,
                "response": str,
                "visualization": Figure,
                "chart_type": str,
                "error": str
            }
        """
        
        try:
            # 🔥 STEP 0: AUTO-DETECT DATABASE DOMAIN
            detected_domain, domain_confidence, all_domain_scores = \
                self.intent_classifier.detect_domain(database_schema)
            
            print(f"\n🎯 Detected Domain: {detected_domain.upper()} (confidence: {domain_confidence:.2%})")
            if domain_confidence < 0.5:
                print(f"   ⚠️  Low confidence - treating as general database")
            
            # Step 1: Classify intent and extract entities
            intent_data = self.intent_classifier.classify(user_prompt)
            entities = self.intent_classifier.extract_entities(user_prompt)
            
            # Add domain info to intent data
            intent_data['entities'] = entities
            intent_data['domain'] = detected_domain
            intent_data['domain_confidence'] = domain_confidence
            intent_data['all_domain_scores'] = all_domain_scores
            
            # Step 2: Generate SQL query (domain-aware)
            generated_query = self.query_generator.generate_sql(
                user_prompt,
                intent_data,
                database_schema
            )
            
            # Step 3: Execute query (if callback provided)
            query_results = None
            if execute_query:
                try:
                    query_results = execute_query(generated_query)
                except Exception as e:
                    return {
                        "success": False,
                        "domain": detected_domain,
                        "domain_confidence": domain_confidence,
                        "intent": intent_data,
                        "generated_query": generated_query,
                        "query_results": None,
                        "visualization": None,
                        "chart_type": "none",
                        "error": f"Query execution failed: {str(e)}",
                        "response": f"I generated a SQL query but couldn't execute it: {str(e)}"
                    }
            
            # Step 4: Generate natural language response (domain-aware)
            response_text = self.response_generator.generate(
                user_prompt,
                query_results,
                intent_data
            )
            
            # Step 5: Generate visualization (domain-aware)
            visualization, chart_type = None, "none"
            if query_results is not None and not query_results.empty:
                visualization, chart_type = self.visualizer.create_chart(
                    query_results,
                    user_prompt,
                    intent_data['intent'],
                    detected_domain  # 🔥 Pass domain for smart chart selection
                )
            
            # Return complete result
            return {
                "success": True,
                "domain": detected_domain,                    # 🔥 NEW
                "domain_confidence": domain_confidence,       # 🔥 NEW
                "all_domain_scores": all_domain_scores,      # 🔥 NEW
                "intent": intent_data,
                "generated_query": generated_query,
                "query_results": query_results,
                "response": response_text,
                "visualization": visualization,
                "chart_type": chart_type
            }
        
        except Exception as e:
            # Handle any errors
            return {
                "success": False,
                "domain": "unknown",
                "error": str(e),
                "response": f"I encountered an error while processing your request: {str(e)}"
            }
    
    def get_supported_domains(self) -> Dict[str, list]:
        """
        Get list of all supported domains and their keywords
        
        Returns:
            dict: {domain_name: [keywords]}
        """
        domains = {}
        for domain, config in self.intent_classifier.domain_signatures.items():
            domains[domain] = config['keywords']
        return domains
    
    def analyze_schema(self, database_schema: Dict) -> Dict[str, Any]:
        """
        Analyze database schema without processing a query
        Useful for debugging and understanding your database
        
        Args:
            database_schema: Database schema dict
            
        Returns:
            dict: {
                "detected_domain": str,
                "confidence": float,
                "all_scores": dict,
                "tables": list,
                "total_columns": int,
                "recommendation": str
            }
        """
        detected_domain, confidence, all_scores = \
            self.intent_classifier.detect_domain(database_schema)
        
        total_columns = sum(len(cols) for cols in database_schema.values())
        
        # Generate recommendation
        if confidence > 0.7:
            recommendation = f"Strong match! This appears to be a {detected_domain} database."
        elif confidence > 0.4:
            recommendation = f"  Moderate match. Likely a {detected_domain} database, but consider adding domain-specific tables/columns."
        else:
            recommendation = f" Low confidence. Schema appears generic. Top candidates: {sorted(all_scores, key=all_scores.get, reverse=True)[:3]}"
        
        return {
            "detected_domain": detected_domain,
            "confidence": confidence,
            "all_scores": all_scores,
            "tables": list(database_schema.keys()),
            "total_columns": total_columns,
            "recommendation": recommendation
        }