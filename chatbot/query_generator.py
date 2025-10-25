"""
Domain-Aware SQL Query Generator using LLM
"""

from typing import Dict
from .llm_manager import FreeLLMManager


class QueryGenerator:
    """Generates SQL queries with domain awareness"""
    
    def __init__(self):
        self.llm = FreeLLMManager()
        
        # Domain-specific SQL tips
        self.domain_sql_tips = {
            'healthcare': [
                "Use patient_id for joins with patient tables",
                "Consider HIPAA compliance - don't expose sensitive fields without permission",
                "Common metrics: patient count, diagnosis frequency, treatment outcomes",
                "Date fields often: admission_date, discharge_date, appointment_date"
            ],
            'finance': [
                "Always use DECIMAL for monetary values, not FLOAT",
                "Common aggregations: SUM(amount), AVG(balance), COUNT(transactions)",
                "Consider currency conversions if multi-currency",
                "Date fields often: transaction_date, payment_date, due_date"
            ],
            'hospital': [
                "Use bed_id, ward_id for hospital operations",
                "Track admission/discharge dates carefully",
                "Common metrics: bed occupancy, average stay duration, department load",
                "Emergency cases may need priority flagging"
            ],
            'retail': [
                "Use product_id, customer_id, order_id for relationships",
                "Inventory tracking: stock levels, reorder points",
                "Common metrics: total sales, items sold, average order value",
                "Consider seasonal trends in date filters"
            ],
            'education': [
                "Use student_id, course_id, teacher_id for relationships",
                "GPA calculations may need weighted averages",
                "Common metrics: enrollment count, average grades, attendance rate",
                "Academic calendars: semesters, terms, academic years"
            ],
            'hr': [
                "Use employee_id, department_id for relationships",
                "Salary data is sensitive - ensure access control",
                "Common metrics: headcount, average salary, turnover rate",
                "Date fields: hire_date, termination_date, appraisal_date"
            ],
            'logistics': [
                "Track shipment_id, delivery_id, warehouse_id",
                "Location data: origin, destination, current_location",
                "Common metrics: delivery time, routes efficiency, capacity utilization",
                "Status tracking: pending, in_transit, delivered"
            ],
            'ecommerce': [
                "Use order_id, product_id, customer_id, cart_id",
                "Track order status: pending, confirmed, shipped, delivered, cancelled",
                "Common metrics: conversion rate, cart abandonment, average order value",
                "Consider user sessions and browsing behavior"
            ]
        }
    
    def generate_sql(
        self,
        user_prompt: str,
        intent_data: Dict,
        schema: Dict
    ) -> str:
        """Generate domain-aware SQL query"""
        
        # Get domain from intent data
        domain = intent_data.get('domain', 'general')
        
        # Build enhanced prompt with domain context
        prompt = self._build_domain_aware_prompt(
            user_prompt,
            intent_data,
            schema,
            domain
        )
        
        messages = [
            {
                "role": "system",
                "content": f"You are a SQL expert specializing in {domain} databases. Generate ONLY valid SQL queries without explanations."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        # Generate SQL using LLM
        sql = self.llm.generate(messages, temperature=0.1, max_tokens=512)
        
        # Clean up response
        sql = self._clean_sql(sql)
        
        return sql
    
    def _build_domain_aware_prompt(
        self,
        user_prompt: str,
        intent_data: Dict,
        schema: Dict,
        domain: str
    ) -> str:
        """Build prompt with domain-specific context"""
        
        entities = intent_data.get('entities', {})
        domain_tips = self.domain_sql_tips.get(domain, [])
        
        prompt = f"""Generate a SQL query for this request:

USER REQUEST: {user_prompt}

DETECTED DOMAIN: {domain.upper()}
INTENT: {intent_data['intent']}
METRICS: {entities.get('metrics', [])}
DIMENSIONS: {entities.get('dimensions', [])}
TIME PERIOD: {entities.get('time_period', 'all')}
AGGREGATION: {entities.get('aggregation', 'none')}
LIMIT: {entities.get('limit', 'none')}

DATABASE SCHEMA:
{self._format_schema(schema)}

DOMAIN-SPECIFIC GUIDELINES ({domain}):
{self._format_domain_tips(domain_tips)}

INSTRUCTIONS:
1. Generate ONLY the SQL query (no explanations)
2. No markdown formatting or code blocks
3. Use proper JOINs if multiple tables needed
4. Add WHERE clause for time filters if specified
5. Use GROUP BY for aggregations
6. Add ORDER BY and LIMIT if specified
7. Follow {domain} domain best practices
8. Ensure query is compatible with PostgreSQL/MySQL/SQLite
9. Use appropriate field names based on domain context

SQL QUERY:"""
        
        return prompt
    
    def _format_schema(self, schema: Dict) -> str:
        """Format database schema for prompt"""
        
        formatted = ""
        for table_name, columns in schema.items():
            formatted += f"\nTable: {table_name}\n"
            formatted += f"Columns: {', '.join(columns)}\n"
        
        return formatted
    
    def _format_domain_tips(self, tips: list) -> str:
        """Format domain-specific tips"""
        if not tips:
            return "- Use standard SQL best practices"
        
        return "\n".join([f"- {tip}" for tip in tips])
    
    def _clean_sql(self, sql: str) -> str:
        """Clean up generated SQL"""
        
        # Remove markdown code blocks
        sql = sql.replace("```sql", "").replace("```", "")
        
        # Remove extra whitespace
        sql = sql.strip()
        
        # Extract SQL if there's text before SELECT
        sql_upper = sql.upper()
        if "SELECT" in sql_upper:
            select_pos = sql_upper.find("SELECT")
            sql = sql[select_pos:]
        
        # Remove any trailing semicolon
        sql = sql.rstrip(';')
        
        return sql