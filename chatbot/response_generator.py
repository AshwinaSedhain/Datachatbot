"""
Natural Language Response Generator
"""

from typing import Dict, Any
from .llm_manager import FreeLLMManager


class ResponseGenerator:
    """Generates natural language responses from query results"""
    
    def __init__(self):
        self.llm = FreeLLMManager()
    
    def generate(
        self,
        user_prompt: str,
        query_results: Any,
        intent_data: Dict
    ) -> str:
        """Generate natural language response in paragraph format"""
        
        # Format results for LLM
        results_text = self._format_results(query_results)
        
        # Build prompt
        prompt = f"""You are a professional business analyst. Provide clear insights in PARAGRAPH format.

USER QUESTION: {user_prompt}

DETECTED INTENT: {intent_data['intent']}

QUERY RESULTS:
{results_text}

INSTRUCTIONS:
1. Write 2-3 well-structured paragraphs
2. First paragraph: Direct answer with key numbers and percentages
3. Second paragraph: Analysis, trends, and detailed breakdown
4. Third paragraph: Key insights and actionable recommendations (if applicable)
5. Use professional business language
6. Include specific metrics and percentages
7. Be concise but comprehensive
8. DO NOT mention SQL or technical details
9. Focus on business value and insights

RESPONSE:"""
        
        messages = [
            {
                "role": "system",
                "content": "You are a professional business analyst who provides clear, actionable insights in paragraph format."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        # Generate response
        response = self.llm.generate(messages, temperature=0.7, max_tokens=1024)
        
        return response
    
    def _format_results(self, results: Any) -> str:
        """Format query results for LLM"""
        
        if results is None:
            return "No results available (query not executed)"
        
        # pandas DataFrame
        if hasattr(results, 'to_string'):
            return results.head(20).to_string()
        
        # Dictionary
        elif isinstance(results, dict):
            return str(results)
        
        # List
        elif isinstance(results, list):
            return "\n".join([str(item) for item in results[:20]])
        
        # Other
        else:
            return str(results)