"""
Training and Test Data Generator for BIRD Framework

This script generates comprehensive training and test datasets for all three domains:
1. Medical Diagnosis Domain
2. Financial Investment Domain  
3. E-commerce Recommendation Domain

Each dataset includes realistic scenarios with proper factor mappings and outcomes.
"""

import csv
import json
import random
from typing import List, Dict, Tuple
from dataclasses import asdict
from bird_framework import DecisionContext
from domain_examples import PatientData, InvestmentProfile, CustomerProfile

# Set random seed for reproducibility
random.seed(42)

def generate_medical_training_data(num_samples: int = 100) -> List[Tuple[Dict, str]]:
    """Generate medical diagnosis training data."""
    training_data = []
    
    # Define factor ranges and their influence on outcomes
    age_ranges = [(18, 35), (36, 50), (51, 65), (66, 85)]
    symptoms_combinations = [
        ["headache", "fatigue"],
        ["chest pain", "shortness of breath"],
        ["fever", "cough", "sore throat"],
        ["nausea", "vomiting", "abdominal pain"],
        ["dizziness", "weakness"],
        ["joint pain", "swelling"],
        ["rash", "itching"],
        ["back pain", "muscle stiffness"],
        ["anxiety", "palpitations"],
        ["insomnia", "mood changes"]
    ]
    
    medical_histories = [
        ["hypertension"],
        ["diabetes"],
        ["asthma"],
        ["heart disease"],
        ["arthritis"],
        ["depression"],
        ["allergies"],
        ["hypertension", "diabetes"],
        ["heart disease", "high cholesterol"],
        []  # No history
    ]
    
    for i in range(num_samples):
        # Generate patient data
        age_range = random.choice(age_ranges)
        age = random.randint(age_range[0], age_range[1])
        gender = random.choice(["male", "female"])
        symptoms = random.choice(symptoms_combinations)
        history = random.choice(medical_histories)
        
        # Generate vital signs
        bp_systolic = random.randint(90, 180)
        bp_diastolic = random.randint(60, 110)
        heart_rate = random.randint(60, 120)
        temperature = round(random.uniform(97.0, 102.0), 1)
        
        vital_signs = {
            "blood_pressure": f"{bp_systolic}/{bp_diastolic}",
            "heart_rate": heart_rate,
            "temperature": temperature
        }
        
        # Generate condition severity and urgency
        severity_indicators = [
            "mild symptoms that started gradually",
            "moderate symptoms present for several days",
            "severe symptoms that started suddenly",
            "worsening symptoms over the past week",
            "acute onset with significant distress",
            "chronic symptoms with recent exacerbation"
        ]
        
        condition = random.choice(severity_indicators)
        
        # Decision logic for conservative vs aggressive treatment
        conservative_factors = 0
        aggressive_factors = 0
        
        # Age factor
        if age < 40:
            conservative_factors += 1
        elif age > 70:
            aggressive_factors += 1
        
        # Symptom severity
        if "severe" in condition or "acute" in condition or "sudden" in condition:
            aggressive_factors += 2
        elif "mild" in condition or "gradual" in condition:
            conservative_factors += 2
        
        # Vital signs
        if bp_systolic > 160 or heart_rate > 100 or temperature > 101:
            aggressive_factors += 1
        
        # Medical history
        if len(history) >= 2:
            aggressive_factors += 1
        elif len(history) == 0:
            conservative_factors += 1
        
        # High-risk symptoms
        high_risk_symptoms = ["chest pain", "shortness of breath", "severe headache"]
        if any(symptom in symptoms for symptom in high_risk_symptoms):
            aggressive_factors += 2
        
        # Determine outcome
        outcome = "aggressive treatment" if aggressive_factors > conservative_factors else "conservative treatment"
        
        # Create structured data
        patient_factors = {
            "age": age,
            "gender": gender,
            "symptoms": symptoms,
            "medical_history": history,
            "vital_signs": vital_signs,
            "condition_severity": condition,
            "symptom_count": len(symptoms),
            "history_count": len(history),
            "bp_systolic": bp_systolic,
            "heart_rate": heart_rate,
            "temperature": temperature
        }
        
        training_data.append((patient_factors, outcome))
    
    return training_data

def generate_financial_training_data(num_samples: int = 100) -> List[Tuple[Dict, str]]:
    """Generate financial investment training data."""
    training_data = []
    
    age_ranges = [(22, 35), (36, 50), (51, 65), (66, 80)]
    income_ranges = [(30000, 60000), (60000, 100000), (100000, 200000), (200000, 500000)]
    risk_tolerances = ["conservative", "moderate", "aggressive"]
    market_conditions = [
        "bull market with low volatility",
        "bear market with high volatility", 
        "sideways market with uncertainty",
        "recovering market after downturn",
        "volatile market with inflation concerns"
    ]
    
    investment_goals = [
        ["retirement"],
        ["house_down_payment"],
        ["children_education"],
        ["retirement", "house_down_payment"],
        ["retirement", "children_education"],
        ["wealth_building", "retirement"],
        ["emergency_fund", "retirement"]
    ]
    
    for i in range(num_samples):
        # Generate investor profile
        age_range = random.choice(age_ranges)
        age = random.randint(age_range[0], age_range[1])
        
        income_range = random.choice(income_ranges)
        income = random.randint(income_range[0], income_range[1])
        
        net_worth = random.randint(int(income * 0.5), int(income * 5))
        risk_tolerance = random.choice(risk_tolerances)
        goals = random.choice(investment_goals)
        
        # Time horizon based on age and goals
        if "retirement" in goals:
            time_horizon = max(5, 65 - age + random.randint(-5, 5))
        else:
            time_horizon = random.randint(3, 15)
        
        market_condition = random.choice(market_conditions)
        
        # Generate current portfolio
        if risk_tolerance == "conservative":
            stocks = random.randint(20, 50)
            bonds = random.randint(40, 70)
        elif risk_tolerance == "moderate":
            stocks = random.randint(50, 70)
            bonds = random.randint(20, 40)
        else:  # aggressive
            stocks = random.randint(70, 90)
            bonds = random.randint(5, 20)
        
        cash = 100 - stocks - bonds
        
        portfolio = {
            "stocks": stocks,
            "bonds": bonds,
            "cash": max(0, cash)
        }
        
        # Generate market condition
        market_stress_indicators = [
            "market volatility is increasing",
            "economic recession concerns",
            "inflation is rising rapidly",
            "interest rates are climbing",
            "geopolitical tensions affecting markets"
        ]
        
        condition = random.choice(market_stress_indicators)
        
        # Decision logic for growth vs income strategy
        growth_factors = 0
        income_factors = 0
        
        # Age factor
        if age < 40:
            growth_factors += 2
        elif age > 60:
            income_factors += 2
        
        # Risk tolerance
        if risk_tolerance == "aggressive":
            growth_factors += 2
        elif risk_tolerance == "conservative":
            income_factors += 2
        
        # Time horizon
        if time_horizon > 15:
            growth_factors += 1
        elif time_horizon < 5:
            income_factors += 1
        
        # Market conditions
        if "bull" in market_condition or "recovering" in market_condition:
            growth_factors += 1
        elif "bear" in market_condition or "volatile" in market_condition:
            income_factors += 1
        
        # Goals
        if "retirement" in goals and age > 55:
            income_factors += 1
        elif "wealth_building" in goals:
            growth_factors += 1
        
        # Current condition
        if "recession" in condition or "volatility" in condition:
            income_factors += 1
        
        outcome = "growth-focused strategy" if growth_factors > income_factors else "income-focused strategy"
        
        # Create structured data
        investor_factors = {
            "age": age,
            "income": income,
            "net_worth": net_worth,
            "risk_tolerance": risk_tolerance,
            "investment_goals": goals,
            "time_horizon": time_horizon,
            "current_portfolio": portfolio,
            "market_conditions": market_condition,
            "current_condition": condition,
            "stocks_percentage": stocks,
            "bonds_percentage": bonds,
            "cash_percentage": cash
        }
        
        training_data.append((investor_factors, outcome))
    
    return training_data

def generate_ecommerce_training_data(num_samples: int = 100) -> List[Tuple[Dict, str]]:
    """Generate e-commerce recommendation training data."""
    training_data = []
    
    age_ranges = [(18, 25), (26, 35), (36, 50), (51, 70)]
    locations = ["urban area", "suburban area", "rural area"]
    income_brackets = ["low income", "middle income", "upper middle income", "high income"]
    
    purchase_categories = [
        ["electronics", "books"],
        ["clothing", "accessories"],
        ["home_goods", "kitchen"],
        ["sports", "fitness"],
        ["beauty", "health"],
        ["electronics", "gaming"],
        ["books", "education"],
        ["travel", "experiences"]
    ]
    
    seasonal_factors = [
        "holiday shopping season",
        "back-to-school season", 
        "summer vacation period",
        "spring cleaning time",
        "winter clothing season",
        "regular shopping period"
    ]
    
    for i in range(num_samples):
        # Generate customer profile
        age_range = random.choice(age_ranges)
        age = random.randint(age_range[0], age_range[1])
        gender = random.choice(["male", "female"])
        location = random.choice(locations)
        income_bracket = random.choice(income_brackets)
        
        # Purchase history based on age and income
        categories = random.choice(purchase_categories)
        purchase_history = []
        for category in categories:
            num_purchases = random.randint(1, 4)
            for _ in range(num_purchases):
                purchase_history.append(f"{category}_item_{random.randint(1, 100)}")
        
        # Browsing behavior
        browsing_behavior = []
        for _ in range(random.randint(3, 8)):
            browsing_behavior.append(f"browsed_{random.choice(categories)}_item_{random.randint(1, 50)}")
        
        seasonal = random.choice(seasonal_factors)
        budget_range = random.choice(["$10-50", "$50-100", "$100-300", "$300-500", "$500+"])
        
        # Generate shopping condition
        shopping_conditions = [
            "looking for a specific item with detailed requirements",
            "browsing casually without specific intent",
            "shopping for a gift with time constraints",
            "comparing prices across multiple options",
            "seeking recommendations for new products",
            "following up on previously viewed items"
        ]
        
        condition = random.choice(shopping_conditions)
        
        # Decision logic for personalized vs trending recommendations
        personalized_factors = 0
        trending_factors = 0
        
        # Purchase history richness
        if len(purchase_history) > 5:
            personalized_factors += 2
        elif len(purchase_history) < 2:
            trending_factors += 1
        
        # Age factor
        if age < 30:
            trending_factors += 1
        elif age > 45:
            personalized_factors += 1
        
        # Shopping behavior
        if "specific" in condition or "detailed" in condition:
            personalized_factors += 2
        elif "casual" in condition or "browsing" in condition:
            trending_factors += 2
        
        # Seasonal factor
        if "holiday" in seasonal or "back-to-school" in seasonal:
            trending_factors += 1
        
        # Budget consideration
        if "$500+" in budget_range or "$300-500" in budget_range:
            personalized_factors += 1
        elif "$10-50" in budget_range:
            trending_factors += 1
        
        # Gift shopping
        if "gift" in condition:
            trending_factors += 1
        
        outcome = "personalized recommendations" if personalized_factors > trending_factors else "trending recommendations"
        
        # Create structured data
        customer_factors = {
            "age": age,
            "gender": gender,
            "location": location,
            "income_bracket": income_bracket,
            "purchase_history": purchase_history,
            "browsing_behavior": browsing_behavior,
            "seasonal_factors": seasonal,
            "budget_range": budget_range,
            "shopping_condition": condition,
            "purchase_history_count": len(purchase_history),
            "browsing_count": len(browsing_behavior),
            "primary_categories": categories
        }
        
        training_data.append((customer_factors, outcome))
    
    return training_data

def save_training_data_python(domain: str, training_data: List, test_data: List):
    """Save training data in Python format."""
    filename = f"/home/ubuntu/{domain}_training_data.py"
    
    with open(filename, 'w') as f:
        f.write(f'"""\n{domain.title()} Domain Training and Test Data for BIRD Framework\n"""\n\n')
        f.write("from bird_framework import DecisionContext\n\n")
        
        # Training data
        f.write(f"{domain}_training_data = [\n")
        for factors, outcome in training_data:
            f.write(f"    ({factors}, \"{outcome}\"),\n")
        f.write("]\n\n")
        
        # Test data
        f.write(f"{domain}_test_data = [\n")
        for factors, outcome in test_data:
            f.write(f"    ({factors}, \"{outcome}\"),\n")
        f.write("]\n\n")
        
        # Helper functions
        f.write(f"def get_{domain}_training_data():\n")
        f.write(f"    \"\"\"Get {domain} training data as list of (factors_dict, outcome) tuples.\"\"\"\n")
        f.write(f"    return {domain}_training_data\n\n")
        
        f.write(f"def get_{domain}_test_data():\n")
        f.write(f"    \"\"\"Get {domain} test data as list of (factors_dict, outcome) tuples.\"\"\"\n")
        f.write(f"    return {domain}_test_data\n\n")
        
        f.write(f"def convert_to_decision_contexts(data_list):\n")
        f.write(f"    \"\"\"Convert factor dictionaries to DecisionContext objects.\"\"\"\n")
        f.write(f"    contexts = []\n")
        f.write(f"    for factors, outcome in data_list:\n")
        f.write(f"        # Create scenario and condition from factors\n")
        f.write(f"        scenario = \"Decision scenario based on provided factors\"\n")
        f.write(f"        condition = \"Condition derived from factor values\"\n")
        f.write(f"        outcomes = [\"conservative treatment\", \"aggressive treatment\"] if \"{domain}\" == \"medical\" else \\\n")
        f.write(f"                  [\"growth-focused strategy\", \"income-focused strategy\"] if \"{domain}\" == \"financial\" else \\\n")
        f.write(f"                  [\"personalized recommendations\", \"trending recommendations\"]\n")
        f.write(f"        \n")
        f.write(f"        context = DecisionContext(\n")
        f.write(f"            scenario=scenario,\n")
        f.write(f"            condition=condition,\n")
        f.write(f"            outcomes=outcomes\n")
        f.write(f"        )\n")
        f.write(f"        contexts.append((context, outcome))\n")
        f.write(f"    return contexts\n")

def save_training_data_csv(domain: str, training_data: List, test_data: List):
    """Save training data in CSV format."""
    
    # Flatten the data for CSV
    def flatten_data(data_list):
        flattened = []
        for factors, outcome in data_list:
            row = {"outcome": outcome}
            
            # Flatten nested dictionaries and lists
            for key, value in factors.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        row[f"{key}_{subkey}"] = subvalue
                elif isinstance(value, list):
                    row[f"{key}_count"] = len(value)
                    row[f"{key}_items"] = "|".join(map(str, value))
                else:
                    row[key] = value
            
            flattened.append(row)
        return flattened
    
    # Training data CSV
    training_filename = f"/home/ubuntu/{domain}_training_data.csv"
    training_flattened = flatten_data(training_data)
    
    if training_flattened:
        with open(training_filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=training_flattened[0].keys())
            writer.writeheader()
            writer.writerows(training_flattened)
    
    # Test data CSV
    test_filename = f"/home/ubuntu/{domain}_test_data.csv"
    test_flattened = flatten_data(test_data)
    
    if test_flattened:
        with open(test_filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=test_flattened[0].keys())
            writer.writeheader()
            writer.writerows(test_flattened)

def generate_all_datasets():
    """Generate all training and test datasets."""
    print("Generating comprehensive training and test datasets for BIRD framework...")
    
    domains = {
        "medical": generate_medical_training_data,
        "financial": generate_financial_training_data,
        "ecommerce": generate_ecommerce_training_data
    }
    
    all_datasets = {}
    
    for domain_name, generator_func in domains.items():
        print(f"\nGenerating {domain_name} domain data...")
        
        # Generate training data (100+ samples)
        training_data = generator_func(120)  # Generate extra for variety
        
        # Split into training and test sets
        random.shuffle(training_data)
        train_split = 100
        test_split = 20
        
        final_training = training_data[:train_split]
        final_test = training_data[train_split:train_split + test_split]
        
        print(f"  Training samples: {len(final_training)}")
        print(f"  Test samples: {len(final_test)}")
        
        # Save in both formats
        save_training_data_python(domain_name, final_training, final_test)
        save_training_data_csv(domain_name, final_training, final_test)
        
        all_datasets[domain_name] = {
            "training": final_training,
            "test": final_test
        }
        
        print(f"  Saved: {domain_name}_training_data.py and .csv")
        print(f"  Saved: {domain_name}_test_data.csv")
    
    # Generate combined dataset
    print(f"\nGenerating combined dataset...")
    combined_training = []
    combined_test = []
    
    for domain_name, dataset in all_datasets.items():
        # Add domain identifier to each sample
        for factors, outcome in dataset["training"]:
            factors_with_domain = factors.copy()
            factors_with_domain["domain"] = domain_name
            combined_training.append((factors_with_domain, outcome))
        
        for factors, outcome in dataset["test"]:
            factors_with_domain = factors.copy()
            factors_with_domain["domain"] = domain_name
            combined_test.append((factors_with_domain, outcome))
    
    save_training_data_python("combined", combined_training, combined_test)
    save_training_data_csv("combined", combined_training, combined_test)
    
    print(f"  Combined training samples: {len(combined_training)}")
    print(f"  Combined test samples: {len(combined_test)}")
    
    return all_datasets

if __name__ == "__main__":
    datasets = generate_all_datasets()
    
    print("\n" + "="*60)
    print("DATASET GENERATION COMPLETE")
    print("="*60)
    
    print("\nGenerated files:")
    domains = ["medical", "financial", "ecommerce", "combined"]
    for domain in domains:
        print(f"  - {domain}_training_data.py")
        print(f"  - {domain}_training_data.csv")
        print(f"  - {domain}_test_data.csv")
    
    print(f"\nTotal datasets: {len(domains)} domains")
    print(f"Training samples per domain: 100")
    print(f"Test samples per domain: 20")
    print(f"Combined training samples: 300")
    print(f"Combined test samples: 60")
    
    print("\nUsage:")
    print("  Python: from medical_training_data import get_medical_training_data")
    print("  CSV: pd.read_csv('medical_training_data.csv')")

