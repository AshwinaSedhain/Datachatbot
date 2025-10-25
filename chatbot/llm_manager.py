"""
FREE LLM Manager - Uses Groq API (Llama 3.1 70B)
100% FREE - No costs
"""

import os
from typing import List, Dict
from groq import Groq


class FreeLLMManager:
    """Manages FREE LLM (Groq - Llama 3.1 70B)"""
    
    def __init__(self):
        groq_key = os.getenv("GROQ_API_KEY")
        
        if not groq_key:
            raise ValueError(
                "\n GROQ_API_KEY not found in environment variables!\n"
                "   Get FREE key from: https://console.groq.com\n"
                "   Then create .env file with: GROQ_API_KEY=your_key_here\n"
            )
        
        try:
            self.client = Groq(api_key=groq_key)
        except Exception as e:
            raise Exception(f"Failed to initialize Groq: {e}")
    
    def generate(
        self,
        messages: List[Dict],
        temperature: float = 0.7,
        max_tokens: int = 1024
    ) -> str:
        """Generate response using FREE Groq API"""
        
        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        
        except Exception as e:
            raise Exception(f"LLM generation failed: {e}")
