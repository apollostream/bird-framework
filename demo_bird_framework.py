"""
Demonstration script for the BIRD framework implementation.

This script demonstrates the framework functionality with the correct model configuration.
"""

import sys
import time
from bird_framework import (
    BirdFramework, BirdConfig, DecisionContext, 
    Factor, FactorValue
)
from domain_examples import (
    DomainFactory, PatientData, InvestmentProfile, CustomerProfile
)

def demo_basic_functionality():
    """Demonstrate basic BIRD framework functionality."""
    print("="*80)
    print("BIRD FRAMEWORK DEMONSTRATION")
    print("="*80)
    
    # Use supported model
    config = BirdConfig(
        llm_model="gpt-4.1-mini",  # Use supported model
        max_factors=3,
        factor_pruning_threshold=0.2,
        temperature=0.3
    )
    
    print(f"Configuration:")
    print(f"- Model: {config.llm_model}")
    print(f"- Max factors: {config.max_factors}")
    print(f"- Pruning threshold: {config.factor_pruning_threshold}")
    
    bird = BirdFramework(config)
    
    # Example 1: Phone charging scenario (from the paper)
    print(f"\n{'-'*60}")
    print("EXAMPLE 1: Phone Charging Decision (from BIRD paper)")
    print(f"{'-'*60}")
    
    context1 = DecisionContext(
        scenario="You want to charge your phone while using it",
        condition="You will walk around the room frequently",
        outcomes=["use a shorter cord", "use a longer cord"]
    )
    
    print(f"Scenario: {context1.scenario}")
    print(f"Condition: {context1.condition}")
    print(f"Outcomes: {context1.outcomes}")
    
    try:
        start_time = time.time()
        result1 = bird.make_decision(context1)
        end_time = time.time()
        
        print(f"\nDecision made in {end_time - start_time:.2f} seconds")
        print(f"Recommended outcome: {result1.get_recommended_outcome()}")
        print(f"Confidence: {result1.confidence:.3f}")
        print(f"Outcome probabilities:")
        for outcome, prob in result1.outcome_probabilities.items():
            print(f"  - {outcome}: {prob:.3f}")
        
        print(f"\nFactor mappings:")
        for factor in result1.factor_mappings:
            print(f"  - {factor.factor_name}: {factor.value} (confidence: {factor.probability:.3f})")
        
        print(f"\nExplanation:")
        print(result1.explanation)
        
    except Exception as e:
        print(f"Error in basic example: {e}")
        return False
    
    # Example 2: Job decision
    print(f"\n{'-'*60}")
    print("EXAMPLE 2: Job Decision")
    print(f"{'-'*60}")
    
    context2 = DecisionContext(
        scenario="You have two job offers and need to choose one",
        condition="Work-life balance is very important to you and you prefer remote work",
        outcomes=["accept job offer A", "accept job offer B"]
    )
    
    print(f"Scenario: {context2.scenario}")
    print(f"Condition: {context2.condition}")
    
    try:
        result2 = bird.make_decision(context2)
        print(f"Recommended outcome: {result2.get_recommended_outcome()}")
        print(f"Confidence: {result2.confidence:.3f}")
        
        print(f"\nKey factors considered:")
        for factor in result2.factor_mappings:
            print(f"  - {factor.factor_name}: {factor.value}")
        
    except Exception as e:
        print(f"Error in job example: {e}")
        return False
    
    return True

def demo_medical_domain():
    """Demonstrate medical domain functionality."""
    print(f"\n{'-'*60}")
    print("MEDICAL DOMAIN DEMONSTRATION")
    print(f"{'-'*60}")
    
    try:
        config = BirdConfig(llm_model="gpt-4.1-mini", max_factors=4)
        bird = BirdFramework(config)
        medical_domain = DomainFactory.create_domain("medical", bird)
        
        # Create patient data
        patient = PatientData(
            age=55,
            gender="female",
            symptoms=["headache", "dizziness", "nausea"],
            medical_history=["hypertension"],
            vital_signs={"blood_pressure": 150, "heart_rate": 88, "temperature": 98.8}
        )
        
        condition = "Symptoms started suddenly this morning and patient has no history of migraines"
        
        print(f"Patient Profile:")
        print(f"- Age: {patient.age}, Gender: {patient.gender}")
        print(f"- Symptoms: {', '.join(patient.symptoms)}")
        print(f"- Medical history: {', '.join(patient.medical_history)}")
        print(f"- Condition: {condition}")
        
        result = medical_domain.make_domain_decision(patient, condition)
        interpretation = medical_domain.interpret_result(result)
        
        print(f"\nMedical Recommendation: {result.get_recommended_outcome()}")
        print(f"Clinical Confidence: {result.confidence:.3f}")
        print(f"\nDetailed Interpretation:")
        print(interpretation[:500] + "..." if len(interpretation) > 500 else interpretation)
        
        return True
        
    except Exception as e:
        print(f"Error in medical domain demo: {e}")
        return False

def demo_framework_components():
    """Demonstrate individual framework components."""
    print(f"\n{'-'*60}")
    print("FRAMEWORK COMPONENTS DEMONSTRATION")
    print(f"{'-'*60}")
    
    try:
        config = BirdConfig(llm_model="gpt-4.1-mini", max_factors=3)
        bird = BirdFramework(config)
        
        # 1. Factor Generation
        print("1. Factor Generation:")
        scenario = "Choosing between two apartments to rent"
        outcomes = ["apartment A", "apartment B"]
        
        factors = bird.factor_generator.generate_factors(scenario, outcomes)
        print(f"Generated {len(factors)} factors for apartment decision:")
        for i, factor in enumerate(factors, 1):
            print(f"   {i}. {factor.name}: {factor.description}")
            print(f"      Values: {factor.possible_values}")
            print(f"      Importance: {factor.importance_score:.3f}")
        
        # 2. Condition Mapping
        print(f"\n2. Condition Mapping:")
        condition = "Budget is tight and commute time is very important"
        factor_mappings = bird.condition_mapper.map_condition_to_factors(condition, factors)
        
        print(f"Condition: {condition}")
        print(f"Factor mappings:")
        for mapping in factor_mappings:
            print(f"   - {mapping.factor_name}: {mapping.value} (confidence: {mapping.probability:.3f})")
        
        # 3. Probability Calculation
        print(f"\n3. Probability Calculation:")
        outcome_probs = bird.bayesian_model.predict_probabilities(factor_mappings, outcomes)
        print(f"Outcome probabilities:")
        for outcome, prob in outcome_probs.items():
            print(f"   - {outcome}: {prob:.3f}")
        
        return True
        
    except Exception as e:
        print(f"Error in components demo: {e}")
        return False

def demo_configuration_options():
    """Demonstrate different configuration options."""
    print(f"\n{'-'*60}")
    print("CONFIGURATION OPTIONS DEMONSTRATION")
    print(f"{'-'*60}")
    
    configs = [
        BirdConfig(
            llm_model="gpt-4.1-mini",
            max_factors=2,
            factor_pruning_threshold=0.3,
            entailment_method="hierarchy"
        ),
        BirdConfig(
            llm_model="gpt-4.1-mini",
            max_factors=5,
            factor_pruning_threshold=0.1,
            entailment_method="direct"
        )
    ]
    
    context = DecisionContext(
        scenario="Choosing a programming language for a new project",
        condition="The project needs to be completed quickly and team expertise is limited",
        outcomes=["Python", "JavaScript"]
    )
    
    for i, config in enumerate(configs, 1):
        print(f"\nConfiguration {i}:")
        print(f"- Max factors: {config.max_factors}")
        print(f"- Pruning threshold: {config.factor_pruning_threshold}")
        print(f"- Entailment method: {config.entailment_method}")
        
        try:
            bird = BirdFramework(config)
            result = bird.make_decision(context)
            
            print(f"Result: {result.get_recommended_outcome()} (confidence: {result.confidence:.3f})")
            print(f"Factors considered: {len(result.factor_mappings)}")
            
        except Exception as e:
            print(f"Error with config {i}: {e}")
    
    return True

def main():
    """Run all demonstrations."""
    print("Starting BIRD Framework Demonstration...")
    
    demos = [
        ("Basic Functionality", demo_basic_functionality),
        ("Medical Domain", demo_medical_domain),
        ("Framework Components", demo_framework_components),
        ("Configuration Options", demo_configuration_options)
    ]
    
    results = {}
    total_start = time.time()
    
    for demo_name, demo_func in demos:
        try:
            print(f"\n{'='*20} {demo_name} {'='*20}")
            results[demo_name] = demo_func()
        except Exception as e:
            print(f"Demo '{demo_name}' failed: {e}")
            results[demo_name] = False
    
    total_end = time.time()
    
    # Summary
    print(f"\n{'='*80}")
    print("DEMONSTRATION SUMMARY")
    print(f"{'='*80}")
    
    successful = sum(1 for result in results.values() if result)
    total = len(results)
    
    for demo_name, result in results.items():
        status = "SUCCESS" if result else "FAILED"
        print(f"{demo_name:<25}: {status}")
    
    print(f"\nOverall: {successful}/{total} demonstrations successful")
    print(f"Total execution time: {total_end - total_start:.2f} seconds")
    
    if successful == total:
        print("\n🎉 All demonstrations completed successfully!")
        print("The BIRD framework is working correctly.")
    else:
        print(f"\n⚠️  {total - successful} demonstration(s) failed.")
    
    return successful == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

