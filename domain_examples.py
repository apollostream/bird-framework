"""
Domain-specific implementations for the BIRD framework.

This module contains three example domains:
1. Medical Diagnosis Domain
2. Financial Investment Domain  
3. Product Recommendation Domain

Each domain demonstrates how to adapt the BIRD framework for specific use cases.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from bird_framework import BaseDomain, BirdFramework, DecisionResult, DecisionContext

# ============================================================================
# Medical Diagnosis Domain
# ============================================================================

@dataclass
class PatientData:
    """Patient data for medical diagnosis."""
    age: int
    gender: str
    symptoms: List[str]
    medical_history: List[str]
    vital_signs: Dict[str, float]
    lab_results: Dict[str, float] = None
    
    def __post_init__(self):
        if self.lab_results is None:
            self.lab_results = {}

class MedicalDiagnosisDomain(BaseDomain):
    """Medical diagnosis decision-making using BIRD framework."""
    
    def __init__(self, bird_framework: BirdFramework):
        super().__init__(bird_framework)
        self.common_diagnoses = [
            "viral infection", "bacterial infection", "allergic reaction",
            "stress-related condition", "chronic condition flare-up",
            "medication side effect", "lifestyle-related issue"
        ]
    
    def format_scenario(self, domain_data: PatientData) -> str:
        """Convert patient data to medical scenario text."""
        scenario = f"A {domain_data.age}-year-old {domain_data.gender} patient presents with "
        
        if domain_data.symptoms:
            scenario += f"symptoms including {', '.join(domain_data.symptoms)}. "
        
        if domain_data.medical_history:
            scenario += f"Medical history includes {', '.join(domain_data.medical_history)}. "
        
        if domain_data.vital_signs:
            vital_info = []
            for sign, value in domain_data.vital_signs.items():
                vital_info.append(f"{sign}: {value}")
            scenario += f"Vital signs show {', '.join(vital_info)}. "
        
        if domain_data.lab_results:
            lab_info = []
            for test, result in domain_data.lab_results.items():
                lab_info.append(f"{test}: {result}")
            scenario += f"Lab results indicate {', '.join(lab_info)}. "
        
        scenario += "The physician needs to decide on the most appropriate initial treatment approach."
        
        return scenario
    
    def define_outcomes(self) -> List[str]:
        """Define medical treatment outcomes."""
        return ["conservative treatment", "aggressive treatment"]
    
    def validate_condition(self, condition: str) -> bool:
        """Validate medical condition input."""
        # Check for basic medical terminology or patient preferences
        medical_keywords = [
            "pain", "severe", "mild", "chronic", "acute", "patient", "family",
            "history", "allergic", "medication", "treatment", "symptoms",
            "urgent", "emergency", "stable", "deteriorating", "improving"
        ]
        
        condition_lower = condition.lower()
        return any(keyword in condition_lower for keyword in medical_keywords) or len(condition) > 10
    
    def interpret_result(self, decision_result: DecisionResult) -> str:
        """Provide medical interpretation of results."""
        recommended = decision_result.get_recommended_outcome()
        confidence = decision_result.confidence
        
        interpretation = f"Medical Recommendation: {recommended.title()}\n"
        interpretation += f"Clinical Confidence: {confidence:.2f}\n\n"
        
        if recommended == "conservative treatment":
            interpretation += "Recommended approach:\n"
            interpretation += "- Start with less invasive interventions\n"
            interpretation += "- Monitor patient response closely\n"
            interpretation += "- Consider lifestyle modifications\n"
            interpretation += "- Schedule follow-up in 1-2 weeks\n"
        else:
            interpretation += "Recommended approach:\n"
            interpretation += "- Implement immediate intervention\n"
            interpretation += "- Consider specialist consultation\n"
            interpretation += "- Monitor for adverse reactions\n"
            interpretation += "- Frequent follow-up required\n"
        
        interpretation += f"\nFactor Analysis:\n"
        for factor in decision_result.factor_mappings:
            interpretation += f"- {factor.factor_name}: {factor.value} (certainty: {factor.probability:.2f})\n"
        
        return interpretation

# ============================================================================
# Financial Investment Domain
# ============================================================================

@dataclass
class InvestmentProfile:
    """Investment profile data."""
    age: int
    income: float
    net_worth: float
    risk_tolerance: str  # "conservative", "moderate", "aggressive"
    investment_goals: List[str]
    time_horizon: int  # years
    current_portfolio: Dict[str, float]  # asset allocation percentages
    market_conditions: str
    
    def __post_init__(self):
        if self.current_portfolio is None:
            self.current_portfolio = {}

class FinancialInvestmentDomain(BaseDomain):
    """Financial investment decision-making using BIRD framework."""
    
    def __init__(self, bird_framework: BirdFramework):
        super().__init__(bird_framework)
        self.asset_classes = ["stocks", "bonds", "real_estate", "commodities", "cash"]
    
    def format_scenario(self, domain_data: InvestmentProfile) -> str:
        """Convert investment profile to financial scenario text."""
        scenario = f"A {domain_data.age}-year-old investor with an annual income of ${domain_data.income:,.0f} "
        scenario += f"and net worth of ${domain_data.net_worth:,.0f} is considering investment options. "
        
        scenario += f"Their risk tolerance is {domain_data.risk_tolerance} and "
        scenario += f"investment time horizon is {domain_data.time_horizon} years. "
        
        if domain_data.investment_goals:
            scenario += f"Investment goals include {', '.join(domain_data.investment_goals)}. "
        
        if domain_data.current_portfolio:
            portfolio_desc = []
            for asset, percentage in domain_data.current_portfolio.items():
                portfolio_desc.append(f"{percentage}% {asset}")
            scenario += f"Current portfolio allocation: {', '.join(portfolio_desc)}. "
        
        scenario += f"Current market conditions are described as {domain_data.market_conditions}. "
        scenario += "The investor needs to decide on their investment strategy."
        
        return scenario
    
    def define_outcomes(self) -> List[str]:
        """Define investment strategy outcomes."""
        return ["growth-focused strategy", "income-focused strategy"]
    
    def validate_condition(self, condition: str) -> bool:
        """Validate investment condition input."""
        financial_keywords = [
            "market", "economy", "inflation", "interest", "rate", "volatility",
            "recession", "growth", "dividend", "capital", "gains", "loss",
            "portfolio", "diversification", "risk", "return", "investment",
            "retirement", "emergency", "fund", "debt", "mortgage", "tax"
        ]
        
        condition_lower = condition.lower()
        return any(keyword in condition_lower for keyword in financial_keywords) or len(condition) > 10
    
    def interpret_result(self, decision_result: DecisionResult) -> str:
        """Provide financial interpretation of results."""
        recommended = decision_result.get_recommended_outcome()
        confidence = decision_result.confidence
        
        interpretation = f"Investment Recommendation: {recommended.title()}\n"
        interpretation += f"Confidence Level: {confidence:.2f}\n\n"
        
        if recommended == "growth-focused strategy":
            interpretation += "Recommended allocation:\n"
            interpretation += "- 70-80% Equities (stocks, growth funds)\n"
            interpretation += "- 15-25% Fixed income (bonds)\n"
            interpretation += "- 5-10% Alternative investments\n"
            interpretation += "- Focus on capital appreciation\n"
            interpretation += "- Higher volatility, higher potential returns\n"
        else:
            interpretation += "Recommended allocation:\n"
            interpretation += "- 40-50% Equities (dividend-paying stocks)\n"
            interpretation += "- 40-50% Fixed income (bonds, REITs)\n"
            interpretation += "- 10% Cash and cash equivalents\n"
            interpretation += "- Focus on regular income generation\n"
            interpretation += "- Lower volatility, steady returns\n"
        
        interpretation += f"\nKey Decision Factors:\n"
        for factor in decision_result.factor_mappings:
            interpretation += f"- {factor.factor_name}: {factor.value} (weight: {factor.probability:.2f})\n"
        
        interpretation += f"\nRisk Assessment:\n"
        if confidence > 0.7:
            interpretation += "- High confidence in recommendation\n"
        elif confidence > 0.5:
            interpretation += "- Moderate confidence, consider diversification\n"
        else:
            interpretation += "- Low confidence, seek professional advice\n"
        
        return interpretation

# ============================================================================
# Product Recommendation Domain
# ============================================================================

@dataclass
class CustomerProfile:
    """Customer profile for product recommendations."""
    age: int
    gender: str
    location: str
    income_bracket: str
    purchase_history: List[str]
    preferences: Dict[str, str]
    browsing_behavior: List[str]
    seasonal_factors: str
    budget_range: str
    
    def __post_init__(self):
        if self.preferences is None:
            self.preferences = {}
        if self.purchase_history is None:
            self.purchase_history = []
        if self.browsing_behavior is None:
            self.browsing_behavior = []

class ProductRecommendationDomain(BaseDomain):
    """Product recommendation decision-making using BIRD framework."""
    
    def __init__(self, bird_framework: BirdFramework):
        super().__init__(bird_framework)
        self.product_categories = [
            "electronics", "clothing", "home_goods", "books", "sports",
            "beauty", "automotive", "food", "toys", "health"
        ]
    
    def format_scenario(self, domain_data: CustomerProfile) -> str:
        """Convert customer profile to recommendation scenario text."""
        scenario = f"A {domain_data.age}-year-old {domain_data.gender} customer from {domain_data.location} "
        scenario += f"with {domain_data.income_bracket} income is browsing for products. "
        
        if domain_data.purchase_history:
            scenario += f"Previous purchases include {', '.join(domain_data.purchase_history[-3:])}. "
        
        if domain_data.preferences:
            pref_desc = []
            for category, preference in domain_data.preferences.items():
                pref_desc.append(f"{preference} {category}")
            scenario += f"Customer preferences: {', '.join(pref_desc)}. "
        
        if domain_data.browsing_behavior:
            scenario += f"Recent browsing includes {', '.join(domain_data.browsing_behavior[-3:])}. "
        
        scenario += f"Current season/timing: {domain_data.seasonal_factors}. "
        scenario += f"Budget range: {domain_data.budget_range}. "
        scenario += "The system needs to decide on the recommendation approach."
        
        return scenario
    
    def define_outcomes(self) -> List[str]:
        """Define recommendation strategy outcomes."""
        return ["personalized recommendations", "trending recommendations"]
    
    def validate_condition(self, condition: str) -> bool:
        """Validate recommendation condition input."""
        ecommerce_keywords = [
            "shopping", "buy", "purchase", "product", "item", "brand",
            "price", "discount", "sale", "review", "rating", "quality",
            "delivery", "shipping", "return", "gift", "occasion",
            "urgent", "browse", "compare", "feature", "specification"
        ]
        
        condition_lower = condition.lower()
        return any(keyword in condition_lower for keyword in ecommerce_keywords) or len(condition) > 10
    
    def interpret_result(self, decision_result: DecisionResult) -> str:
        """Provide e-commerce interpretation of results."""
        recommended = decision_result.get_recommended_outcome()
        confidence = decision_result.confidence
        
        interpretation = f"Recommendation Strategy: {recommended.title()}\n"
        interpretation += f"Algorithm Confidence: {confidence:.2f}\n\n"
        
        if recommended == "personalized recommendations":
            interpretation += "Recommendation approach:\n"
            interpretation += "- Use customer's purchase history\n"
            interpretation += "- Consider stated preferences\n"
            interpretation += "- Apply collaborative filtering\n"
            interpretation += "- Include similar customer patterns\n"
            interpretation += "- Prioritize relevance over popularity\n"
        else:
            interpretation += "Recommendation approach:\n"
            interpretation += "- Show currently trending items\n"
            interpretation += "- Include seasonal bestsellers\n"
            interpretation += "- Feature new arrivals\n"
            interpretation += "- Apply social proof (reviews, ratings)\n"
            interpretation += "- Prioritize popular items\n"
        
        interpretation += f"\nDecision Factors:\n"
        for factor in decision_result.factor_mappings:
            interpretation += f"- {factor.factor_name}: {factor.value} (strength: {factor.probability:.2f})\n"
        
        interpretation += f"\nImplementation Notes:\n"
        if confidence > 0.7:
            interpretation += "- High confidence: Apply strategy consistently\n"
        elif confidence > 0.5:
            interpretation += "- Moderate confidence: Mix both approaches\n"
        else:
            interpretation += "- Low confidence: A/B test different strategies\n"
        
        return interpretation

# ============================================================================
# Domain Factory and Utilities
# ============================================================================

class DomainFactory:
    """Factory for creating domain-specific BIRD implementations."""
    
    @staticmethod
    def create_domain(domain_type: str, bird_framework: BirdFramework) -> BaseDomain:
        """Create a domain-specific implementation."""
        domain_map = {
            "medical": MedicalDiagnosisDomain,
            "financial": FinancialInvestmentDomain,
            "ecommerce": ProductRecommendationDomain
        }
        
        if domain_type not in domain_map:
            raise ValueError(f"Unknown domain type: {domain_type}. Available: {list(domain_map.keys())}")
        
        return domain_map[domain_type](bird_framework)
    
    @staticmethod
    def list_available_domains() -> List[str]:
        """List all available domain types."""
        return ["medical", "financial", "ecommerce"]

def create_sample_data():
    """Create sample data for each domain for testing."""
    
    # Medical sample data
    medical_sample = PatientData(
        age=45,
        gender="female",
        symptoms=["headache", "fatigue", "mild fever"],
        medical_history=["hypertension", "diabetes"],
        vital_signs={"blood_pressure": 140, "heart_rate": 85, "temperature": 99.2},
        lab_results={"white_blood_cells": 8500, "glucose": 120}
    )
    
    # Financial sample data
    financial_sample = InvestmentProfile(
        age=35,
        income=75000,
        net_worth=150000,
        risk_tolerance="moderate",
        investment_goals=["retirement", "house_down_payment"],
        time_horizon=25,
        current_portfolio={"stocks": 60, "bonds": 30, "cash": 10},
        market_conditions="volatile with inflation concerns"
    )
    
    # E-commerce sample data
    ecommerce_sample = CustomerProfile(
        age=28,
        gender="male",
        location="urban area",
        income_bracket="middle income",
        purchase_history=["laptop", "running_shoes", "coffee_maker"],
        preferences={"electronics": "high-tech", "clothing": "casual"},
        browsing_behavior=["smartphones", "fitness_trackers", "headphones"],
        seasonal_factors="back-to-school season",
        budget_range="$200-500"
    )
    
    return {
        "medical": medical_sample,
        "financial": financial_sample,
        "ecommerce": ecommerce_sample
    }

def run_domain_examples():
    """Run examples for all three domains."""
    from bird_framework import BirdFramework, BirdConfig
    
    # Initialize BIRD framework
    config = BirdConfig(llm_model="gpt-3.5-turbo")
    bird = BirdFramework(config)
    
    # Get sample data
    sample_data = create_sample_data()
    
    # Test conditions for each domain
    test_conditions = {
        "medical": "Patient reports worsening symptoms over the past 24 hours and has a family history of autoimmune conditions",
        "financial": "Market volatility is high and the investor is concerned about upcoming recession",
        "ecommerce": "Customer is shopping for a gift and needs fast delivery within 2 days"
    }
    
    results = {}
    
    for domain_type in ["medical", "financial", "ecommerce"]:
        print(f"\n{'='*60}")
        print(f"Testing {domain_type.upper()} Domain")
        print(f"{'='*60}")
        
        try:
            # Create domain instance
            domain = DomainFactory.create_domain(domain_type, bird)
            
            # Make decision
            result = domain.make_domain_decision(
                sample_data[domain_type], 
                test_conditions[domain_type]
            )
            
            # Interpret results
            interpretation = domain.interpret_result(result)
            
            print(f"Scenario: {domain.format_scenario(sample_data[domain_type])[:200]}...")
            print(f"\nCondition: {test_conditions[domain_type]}")
            print(f"\n{interpretation}")
            
            results[domain_type] = {
                "result": result,
                "interpretation": interpretation
            }
            
        except Exception as e:
            print(f"Error in {domain_type} domain: {e}")
            results[domain_type] = {"error": str(e)}
    
    return results

if __name__ == "__main__":
    # Run examples for all domains
    results = run_domain_examples()
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    for domain_type, result in results.items():
        if "error" in result:
            print(f"{domain_type.title()}: Failed - {result['error']}")
        else:
            decision_result = result["result"]
            recommended = decision_result.get_recommended_outcome()
            confidence = decision_result.confidence
            print(f"{domain_type.title()}: {recommended} (confidence: {confidence:.2f})")

