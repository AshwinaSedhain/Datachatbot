"""
Quick Test Script with Domain Detection
Run: python quick_test.py
"""

from dotenv import load_dotenv
load_dotenv()

from chatbot import ChatbotAgent
import pandas as pd


def main():
    print("\n" + "="*80)
    print(" DOMAIN-AWARE AI CHATBOT - COMPREHENSIVE TEST")
    print("="*80)
    
    # Initialize chatbot
    chatbot = ChatbotAgent()
    
    # ============================================
    # TEST 1: HEALTHCARE DATABASE
    # ============================================
    print("\n" + "="*80)
    print("TEST 1: HEALTHCARE DATABASE")
    print("="*80)
    
    healthcare_schema = {
        "patients": ["id", "name", "age", "diagnosis", "admission_date", "blood_type"],
        "doctors": ["id", "name", "specialty", "department"],
        "prescriptions": ["id", "patient_id", "medication", "dosage", "frequency"],
        "appointments": ["id", "patient_id", "doctor_id", "date", "status"]
    }
    
    # Analyze schema
    analysis = chatbot.analyze_schema(healthcare_schema)
    print(f"\n Schema Analysis:")
    print(f"   Detected Domain: {analysis['detected_domain'].upper()}")
    print(f"   Confidence: {analysis['confidence']:.0%}")
    print(f"   Tables: {', '.join(analysis['tables'])}")
    print(f"   Total Columns: {analysis['total_columns']}")
    print(f"   {analysis['recommendation']}")
    
    # Mock query executor
    def mock_healthcare_query(sql):
        print(f"\n   [DB] Executing: {sql[:80]}...")
        return pd.DataFrame({
            'diagnosis': ['Diabetes', 'Hypertension', 'Asthma', 'Heart Disease'],
            'patient_count': [145, 98, 67, 52],
            'avg_age': [58, 62, 45, 65]
        })
    
    # Test question
    question = "Show me patient count by diagnosis"
    print(f"\n Question: {question}")
    
    result = chatbot.process(
        user_prompt=question,
        database_schema=healthcare_schema,
        execute_query=mock_healthcare_query
    )
    
    print_result(result, 1)
    
    # ============================================
    # TEST 2: FINANCE DATABASE
    # ============================================
    print("\n" + "="*80)
    print("TEST 2: FINANCE DATABASE")
    print("="*80)
    
    finance_schema = {
        "transactions": ["id", "date", "amount", "type", "account_id", "balance"],
        "accounts": ["id", "account_number", "balance", "account_type", "customer_id"],
        "invoices": ["id", "invoice_number", "total", "paid", "due_date"],
        "ledger": ["id", "date", "debit", "credit", "description"]
    }
    
    analysis = chatbot.analyze_schema(finance_schema)
    print(f"\n Schema Analysis:")
    print(f"   Detected Domain: {analysis['detected_domain'].upper()}")
    print(f"   Confidence: {analysis['confidence']:.0%}")
    print(f"   {analysis['recommendation']}")
    
    def mock_finance_query(sql):
        print(f"\n   [DB] Executing: {sql[:80]}...")
        return pd.DataFrame({
            'month': ['January', 'February', 'March', 'April'],
            'revenue': [125000, 142000, 138000, 155000],
            'expenses': [85000, 92000, 88000, 95000],
            'profit': [40000, 50000, 50000, 60000]
        })
    
    question = "Show me profit breakdown by month"
    print(f"\n Question: {question}")
    
    result = chatbot.process(
        user_prompt=question,
        database_schema=finance_schema,
        execute_query=mock_finance_query
    )
    
    print_result(result, 2)
    
    # ============================================
    # TEST 3: RETAIL/E-COMMERCE DATABASE
    # ============================================
    print("\n" + "="*80)
    print("TEST 3: RETAIL/E-COMMERCE DATABASE")
    print("="*80)
    
    retail_schema = {
        "products": ["id", "name", "category", "price", "stock", "sku"],
        "orders": ["id", "customer_id", "product_id", "quantity", "total", "date"],
        "customers": ["id", "name", "email", "region", "total_purchases"],
        "inventory": ["id", "product_id", "warehouse", "quantity", "reorder_level"]
    }
    
    analysis = chatbot.analyze_schema(retail_schema)
    print(f"\n🔍 Schema Analysis:")
    print(f"   Detected Domain: {analysis['detected_domain'].upper()}")
    print(f"   Confidence: {analysis['confidence']:.0%}")
    print(f"   {analysis['recommendation']}")
    
    def mock_retail_query(sql):
        print(f"\n   [DB] Executing: {sql[:80]}...")
        return pd.DataFrame({
            'product': ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Headphones'],
            'sales': [45000, 12000, 8000, 25000, 15000],
            'quantity_sold': [150, 800, 500, 200, 600]
        })
    
    question = "What are the top 5 selling products?"
    print(f"\n Question: {question}")
    
    result = chatbot.process(
        user_prompt=question,
        database_schema=retail_schema,
        execute_query=mock_retail_query
    )
    
    print_result(result, 3)
    
    # ============================================
    # TEST 4: EDUCATION DATABASE
    # ============================================
    print("\n" + "="*80)
    print("TEST 4: EDUCATION DATABASE")
    print("="*80)
    
    education_schema = {
        "students": ["id", "name", "email", "enrollment_date", "gpa", "major"],
        "courses": ["id", "name", "code", "credits", "department"],
        "enrollments": ["id", "student_id", "course_id", "grade", "semester"],
        "teachers": ["id", "name", "department", "specialization"]
    }
    
    analysis = chatbot.analyze_schema(education_schema)
    print(f"\n Schema Analysis:")
    print(f"   Detected Domain: {analysis['detected_domain'].upper()}")
    print(f"   Confidence: {analysis['confidence']:.0%}")
    print(f"   {analysis['recommendation']}")
    
    def mock_education_query(sql):
        print(f"\n   [DB] Executing: {sql[:80]}...")
        return pd.DataFrame({
            'department': ['Computer Science', 'Business', 'Engineering', 'Arts'],
            'enrollment': [450, 380, 290, 220],
            'avg_gpa': [3.4, 3.2, 3.5, 3.1]
        })
    
    question = "Show enrollment by department"
    print(f"\n Question: {question}")
    
    result = chatbot.process(
        user_prompt=question,
        database_schema=education_schema,
        execute_query=mock_education_query
    )
    
    print_result(result, 4)
    
    # ============================================
    # TEST 5: HR DATABASE
    # ============================================
    print("\n" + "="*80)
    print("TEST 5: HR DATABASE")
    print("="*80)
    
    hr_schema = {
        "employees": ["id", "name", "email", "department", "salary", "hire_date"],
        "departments": ["id", "name", "manager_id", "budget"],
        "payroll": ["id", "employee_id", "amount", "date", "deductions"],
        "performance": ["id", "employee_id", "rating", "review_date", "bonus"]
    }
    
    analysis = chatbot.analyze_schema(hr_schema)
    print(f"\n Schema Analysis:")
    print(f"   Detected Domain: {analysis['detected_domain'].upper()}")
    print(f"   Confidence: {analysis['confidence']:.0%}")
    print(f"   {analysis['recommendation']}")
    
    def mock_hr_query(sql):
        print(f"\n   [DB] Executing: {sql[:80]}...")
        return pd.DataFrame({
            'department': ['Engineering', 'Sales', 'Marketing', 'HR', 'Operations'],
            'headcount': [45, 32, 18, 8, 25],
            'avg_salary': [95000, 75000, 68000, 72000, 65000]
        })
    
    question = "Show employee headcount and average salary by department"
    print(f"\n  Question: {question}")
    
    result = chatbot.process(
        user_prompt=question,
        database_schema=hr_schema,
        execute_query=mock_hr_query
    )
    
    print_result(result, 5)
    
    # ============================================
    # DOMAIN SUPPORT SUMMARY
    # ============================================
    print("\n" + "="*80)
    print(" SUPPORTED DOMAINS")
    print("="*80)
    
    supported_domains = chatbot.get_supported_domains()
    
    for i, (domain, keywords) in enumerate(supported_domains.items(), 1):
        print(f"\n{i}. {domain.upper()}")
        print(f"   Keywords: {', '.join(keywords[:8])}{'...' if len(keywords) > 8 else ''}")
    
    # ============================================
    # FINAL SUMMARY
    # ============================================
    print("\n" + "="*80)
    print(" ALL TESTS COMPLETED!")
    print("="*80)
    print("\n Key Features Demonstrated:")
    print("  Automatic domain detection (8+ domains)")
    print("  Domain-aware SQL generation")
    print("  Domain-specific response styling")
    print("  Domain-based visualization selection")
    print("  Confidence scoring for domain detection")
    print("\n Integration Guide:")
    print("   1. from chatbot import ChatbotAgent")
    print("   2. chatbot = ChatbotAgent()")
    print("   3. result = chatbot.process(question, schema, execute_query)")
    print("   4. Access: result['domain'], result['response'], result['visualization']")
    print("\n Supported Domains:")
    print("   Healthcare | Finance | Hospital | Retail | Education")
    print("   HR | Logistics | E-commerce | General")
    print("\n")


def print_result(result, test_num):
    """Helper function to print results"""
    if result['success']:
        print(f"\n  SUCCESS")
        print(f"\n  Detected Domain: {result['domain'].upper()}")
        print(f"   Confidence: {result['domain_confidence']:.0%}")
        print(f"\n  Intent: {result['intent']['intent']}")
        print(f"   Confidence: {result['intent']['confidence']:.2%}")
        print(f"\n  Generated SQL:")
        print(f"   {result['generated_query'][:150]}...")
        print(f"\n AI Response:")
        response_preview = result['response'][:400]
        print(f"   {response_preview}...")
        
        if result.get('visualization'):
            print(f"\n  Visualization: {result['chart_type'].upper()} chart generated")
            filename = f"test_{test_num}_{result['domain']}_{result['chart_type']}.html"
            result['visualization'].write_html(filename)
            print(f"   Saved as: {filename}")
        else:
            print(f"\n Visualization: None")
        
        # Show all domain scores
        all_scores = result.get('all_domain_scores', {})
        if all_scores:
            print(f"\n All Domain Scores:")
            sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
            for domain, score in sorted_scores[:5]:
                bar = " " * int(score * 20)
                print(f"   {domain:12s} {score:.0%} {bar}")
    else:
        print(f"\n FAILED")
        print(f"   Error: {result.get('error', 'Unknown error')}")
    
    print("\n" + "─"*80)
    input("   Press Enter to continue...")


if __name__ == "__main__":
    main()