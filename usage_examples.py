"""
Practical usage examples for the BIRD framework.

This script demonstrates how to use the BIRD framework for various
decision-making scenarios across different domains.
"""

from bird_framework import BirdFramework, BirdConfig, DecisionContext
from domain_examples import (
    DomainFactory, PatientData, InvestmentProfile, CustomerProfile
)

def example_1_basic_decision():
    """Example 1: Basic decision-making with BIRD framework."""
    print("="*60)
    print("EXAMPLE 1: Basic Decision Making")
    print("="*60)
    
    # Initialize framework with recommended settings
    config = BirdConfig(
        llm_model="gpt-4.1-mini",
        max_factors=5,
        factor_pruning_threshold=0.2,
        entailment_method="hierarchy"
    )
    bird = BirdFramework(config)
    
    # Example: Choosing a vacation destination
    context = DecisionContext(
        scenario="You are planning a week-long vacation and need to choose a destination",
        condition="You have a limited budget and prefer warm weather with outdoor activities",
        outcomes=["beach destination", "mountain destination"]
    )
    
    print(f"Scenario: {context.scenario}")
    print(f"Condition: {context.condition}")
    print(f"Outcomes: {context.outcomes}")
    
    try:
        result = bird.make_decision(context)
        
        print(f"\n🎯 RECOMMENDATION: {result.get_recommended_outcome()}")
        print(f"📊 Confidence: {result.confidence:.3f}")
        
        print(f"\n📈 Outcome Probabilities:")
        for outcome, prob in result.outcome_probabilities.items():
            print(f"   • {outcome}: {prob:.3f}")
        
        if result.factor_mappings:
            print(f"\n🔍 Key Factors Considered:")
            for factor in result.factor_mappings:
                print(f"   • {factor.factor_name}: {factor.value} (confidence: {factor.probability:.3f})")
        
        print(f"\n💡 Explanation:")
        print(result.explanation)
        
    except Exception as e:
        print(f"❌ Error: {e}")

def example_2_medical_diagnosis():
    """Example 2: Medical diagnosis domain."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Medical Diagnosis Domain")
    print("="*60)
    
    # Initialize framework and medical domain
    config = BirdConfig(llm_model="gpt-4.1-mini", max_factors=4)
    bird = BirdFramework(config)
    medical_domain = DomainFactory.create_domain("medical", bird)
    
    # Create patient scenario
    patient = PatientData(
        age=62,
        gender="male",
        symptoms=["chest pain", "shortness of breath", "sweating"],
        medical_history=["diabetes", "high cholesterol"],
        vital_signs={
            "blood_pressure": 160,
            "heart_rate": 95,
            "temperature": 98.6,
            "oxygen_saturation": 94
        },
        lab_results={
            "troponin": 0.5,
            "creatinine": 1.1
        }
    )
    
    condition = "Patient arrived via ambulance reporting severe chest pain that started 1 hour ago during physical activity"
    
    print(f"👤 Patient Profile:")
    print(f"   • Age: {patient.age}, Gender: {patient.gender}")
    print(f"   • Symptoms: {', '.join(patient.symptoms)}")
    print(f"   • Medical History: {', '.join(patient.medical_history)}")
    print(f"   • Vital Signs: {patient.vital_signs}")
    print(f"   • Lab Results: {patient.lab_results}")
    print(f"\n🚨 Current Condition: {condition}")
    
    try:
        result = medical_domain.make_domain_decision(patient, condition)
        interpretation = medical_domain.interpret_result(result)
        
        print(f"\n🏥 MEDICAL RECOMMENDATION: {result.get_recommended_outcome()}")
        print(f"📊 Clinical Confidence: {result.confidence:.3f}")
        
        print(f"\n📋 Clinical Interpretation:")
        print(interpretation)
        
    except Exception as e:
        print(f"❌ Error: {e}")

def example_3_financial_investment():
    """Example 3: Financial investment domain."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Financial Investment Domain")
    print("="*60)
    
    # Initialize framework and financial domain
    config = BirdConfig(llm_model="gpt-4.1-mini", max_factors=4)
    bird = BirdFramework(config)
    financial_domain = DomainFactory.create_domain("financial", bird)
    
    # Create investor profile
    investor = InvestmentProfile(
        age=35,
        income=95000,
        net_worth=250000,
        risk_tolerance="moderate",
        investment_goals=["retirement", "house_down_payment", "children_education"],
        time_horizon=25,
        current_portfolio={
            "stocks": 65,
            "bonds": 25,
            "real_estate": 5,
            "cash": 5
        },
        market_conditions="volatile with rising inflation and interest rate uncertainty"
    )
    
    condition = "Recent market downturn has reduced portfolio value by 15% and investor is concerned about upcoming recession"
    
    print(f"💼 Investor Profile:")
    print(f"   • Age: {investor.age}, Income: ${investor.income:,}")
    print(f"   • Net Worth: ${investor.net_worth:,}")
    print(f"   • Risk Tolerance: {investor.risk_tolerance}")
    print(f"   • Investment Goals: {', '.join(investor.investment_goals)}")
    print(f"   • Time Horizon: {investor.time_horizon} years")
    print(f"   • Current Portfolio: {investor.current_portfolio}")
    print(f"   • Market Conditions: {investor.market_conditions}")
    print(f"\n📉 Current Situation: {condition}")
    
    try:
        result = financial_domain.make_domain_decision(investor, condition)
        interpretation = financial_domain.interpret_result(result)
        
        print(f"\n💰 INVESTMENT RECOMMENDATION: {result.get_recommended_outcome()}")
        print(f"📊 Confidence Level: {result.confidence:.3f}")
        
        print(f"\n📈 Investment Strategy:")
        print(interpretation)
        
    except Exception as e:
        print(f"❌ Error: {e}")

def example_4_ecommerce_recommendation():
    """Example 4: E-commerce recommendation domain."""
    print("\n" + "="*60)
    print("EXAMPLE 4: E-commerce Recommendation Domain")
    print("="*60)
    
    # Initialize framework and e-commerce domain
    config = BirdConfig(llm_model="gpt-4.1-mini", max_factors=4)
    bird = BirdFramework(config)
    ecommerce_domain = DomainFactory.create_domain("ecommerce", bird)
    
    # Create customer profile
    customer = CustomerProfile(
        age=28,
        gender="female",
        location="urban area",
        income_bracket="upper middle income",
        purchase_history=[
            "wireless_headphones", "fitness_tracker", "laptop", 
            "running_shoes", "coffee_maker", "smartphone"
        ],
        preferences={
            "electronics": "latest technology",
            "clothing": "sustainable brands",
            "home": "minimalist design"
        },
        browsing_behavior=[
            "smartwatches", "wireless_earbuds", "yoga_mats", 
            "organic_skincare", "standing_desk"
        ],
        seasonal_factors="holiday shopping season",
        budget_range="$50-200"
    )
    
    condition = "Customer is shopping for a birthday gift for her tech-savvy friend and needs it delivered within 2 days"
    
    print(f"🛍️ Customer Profile:")
    print(f"   • Demographics: {customer.age}-year-old {customer.gender} from {customer.location}")
    print(f"   • Income Bracket: {customer.income_bracket}")
    print(f"   • Recent Purchases: {', '.join(customer.purchase_history[-4:])}")
    print(f"   • Preferences: {customer.preferences}")
    print(f"   • Recent Browsing: {', '.join(customer.browsing_behavior[-4:])}")
    print(f"   • Seasonal Context: {customer.seasonal_factors}")
    print(f"   • Budget Range: {customer.budget_range}")
    print(f"\n🎁 Shopping Context: {condition}")
    
    try:
        result = ecommerce_domain.make_domain_decision(customer, condition)
        interpretation = ecommerce_domain.interpret_result(result)
        
        print(f"\n🎯 RECOMMENDATION STRATEGY: {result.get_recommended_outcome()}")
        print(f"📊 Algorithm Confidence: {result.confidence:.3f}")
        
        print(f"\n🤖 Recommendation Approach:")
        print(interpretation)
        
    except Exception as e:
        print(f"❌ Error: {e}")

def example_5_custom_scenario():
    """Example 5: Custom scenario with manual factor specification."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Custom Scenario with Manual Factors")
    print("="*60)
    
    from bird_framework import Factor
    
    # Initialize framework
    config = BirdConfig(llm_model="gpt-4.1-mini")
    bird = BirdFramework(config)
    
    # Define custom factors
    custom_factors = [
        Factor(
            name="time_constraint",
            description="How much time is available for the decision",
            possible_values=["very_limited", "moderate", "plenty"]
        ),
        Factor(
            name="cost_sensitivity",
            description="How important cost considerations are",
            possible_values=["low", "medium", "high"]
        ),
        Factor(
            name="quality_importance",
            description="How important quality is relative to other factors",
            possible_values=["low", "medium", "high"]
        )
    ]
    
    # Create decision context with predefined factors
    context = DecisionContext(
        scenario="You need to choose between two software solutions for your startup",
        condition="The project deadline is in 2 weeks and budget is very tight",
        outcomes=["open source solution", "commercial solution"],
        factors=custom_factors
    )
    
    print(f"🔧 Custom Scenario:")
    print(f"   • Scenario: {context.scenario}")
    print(f"   • Condition: {context.condition}")
    print(f"   • Outcomes: {context.outcomes}")
    print(f"   • Predefined Factors: {len(context.factors)}")
    
    for factor in context.factors:
        print(f"     - {factor.name}: {factor.description}")
        print(f"       Values: {factor.possible_values}")
    
    try:
        result = bird.make_decision(context)
        
        print(f"\n⚡ RECOMMENDATION: {result.get_recommended_outcome()}")
        print(f"📊 Confidence: {result.confidence:.3f}")
        
        print(f"\n🔍 Factor Analysis:")
        for factor in result.factor_mappings:
            print(f"   • {factor.factor_name}: {factor.value} (confidence: {factor.probability:.3f})")
        
        print(f"\n💭 Decision Reasoning:")
        print(result.explanation)
        
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Run all usage examples."""
    print("🚀 BIRD Framework - Practical Usage Examples")
    print("=" * 80)
    
    examples = [
        ("Basic Decision Making", example_1_basic_decision),
        ("Medical Diagnosis", example_2_medical_diagnosis),
        ("Financial Investment", example_3_financial_investment),
        ("E-commerce Recommendation", example_4_ecommerce_recommendation),
        ("Custom Scenario", example_5_custom_scenario)
    ]
    
    for name, example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\n❌ Example '{name}' failed: {e}")
    
    print("\n" + "="*80)
    print("✅ All examples completed!")
    print("💡 Tip: Modify the scenarios and conditions to test different use cases.")
    print("📚 See README.md for more detailed documentation.")

if __name__ == "__main__":
    main()

