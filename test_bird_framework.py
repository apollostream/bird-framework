"""
Comprehensive test script for the BIRD framework implementation.

This script tests:
1. Core BIRD framework functionality
2. All three domain implementations
3. Error handling and edge cases
4. Performance and reliability
"""

import sys
import time
import traceback
from typing import Dict, Any

# Import our implementations
from bird_framework import (
    BirdFramework, BirdConfig, DecisionContext, 
    Factor, FactorValue, DecisionResult
)
from domain_examples import (
    DomainFactory, create_sample_data,
    PatientData, InvestmentProfile, CustomerProfile
)

def test_core_framework():
    """Test the core BIRD framework functionality."""
    print("Testing Core BIRD Framework...")
    
    try:
        # Initialize framework
        config = BirdConfig(
            llm_model="gpt-3.5-turbo",
            max_factors=5,
            factor_pruning_threshold=0.2,
            temperature=0.3
        )
        bird = BirdFramework(config)
        
        # Test basic decision making
        context = DecisionContext(
            scenario="You want to charge your phone while using it",
            condition="You will walk around the room frequently",
            outcomes=["use a shorter cord", "use a longer cord"]
        )
        
        print(f"Scenario: {context.scenario}")
        print(f"Condition: {context.condition}")
        print(f"Outcomes: {context.outcomes}")
        
        start_time = time.time()
        result = bird.make_decision(context)
        end_time = time.time()
        
        print(f"\nDecision made in {end_time - start_time:.2f} seconds")
        print(f"Recommended outcome: {result.get_recommended_outcome()}")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Number of factors considered: {len(result.factor_mappings)}")
        
        # Validate result structure
        assert isinstance(result.outcome_probabilities, dict)
        assert len(result.outcome_probabilities) == 2
        assert sum(result.outcome_probabilities.values()) > 0.99  # Should sum to ~1
        assert 0 <= result.confidence <= 1
        assert isinstance(result.explanation, str)
        assert len(result.explanation) > 0
        
        print("✓ Core framework test passed")
        return True
        
    except Exception as e:
        print(f"✗ Core framework test failed: {e}")
        traceback.print_exc()
        return False

def test_factor_generation():
    """Test factor generation functionality."""
    print("\nTesting Factor Generation...")
    
    try:
        config = BirdConfig(max_factors=3, factor_pruning_threshold=0.1)
        bird = BirdFramework(config)
        
        scenario = "Choosing between two job offers"
        outcomes = ["accept job A", "accept job B"]
        
        factors = bird.factor_generator.generate_factors(scenario, outcomes)
        
        print(f"Generated {len(factors)} factors:")
        for i, factor in enumerate(factors, 1):
            print(f"  {i}. {factor.name}: {factor.description}")
            print(f"     Values: {factor.possible_values}")
            print(f"     Importance: {factor.importance_score:.3f}")
        
        # Validate factors
        assert len(factors) > 0
        assert len(factors) <= config.max_factors
        
        for factor in factors:
            assert isinstance(factor.name, str)
            assert len(factor.name) > 0
            assert isinstance(factor.description, str)
            assert len(factor.description) > 0
            assert isinstance(factor.possible_values, list)
            assert len(factor.possible_values) >= 2
            assert 0 <= factor.importance_score <= 1
        
        print("✓ Factor generation test passed")
        return True
        
    except Exception as e:
        print(f"✗ Factor generation test failed: {e}")
        traceback.print_exc()
        return False

def test_condition_mapping():
    """Test condition mapping functionality."""
    print("\nTesting Condition Mapping...")
    
    try:
        config = BirdConfig(entailment_method="hierarchy")
        bird = BirdFramework(config)
        
        # Create test factors
        factors = [
            Factor(
                name="urgency",
                description="How urgent the decision is",
                possible_values=["low", "medium", "high"]
            ),
            Factor(
                name="cost",
                description="Financial cost consideration",
                possible_values=["low", "medium", "high"]
            )
        ]
        
        condition = "This is an emergency situation and budget is not a concern"
        
        factor_mappings = bird.condition_mapper.map_condition_to_factors(condition, factors)
        
        print(f"Condition: {condition}")
        print(f"Factor mappings ({len(factor_mappings)}):")
        for mapping in factor_mappings:
            print(f"  {mapping.factor_name}: {mapping.value} (confidence: {mapping.probability:.3f})")
        
        # Validate mappings
        for mapping in factor_mappings:
            assert isinstance(mapping.factor_name, str)
            assert isinstance(mapping.value, str)
            assert 0 <= mapping.probability <= 1
            
            # Check that the value is valid for the factor
            factor = next(f for f in factors if f.name == mapping.factor_name)
            assert mapping.value in factor.possible_values
        
        print("✓ Condition mapping test passed")
        return True
        
    except Exception as e:
        print(f"✗ Condition mapping test failed: {e}")
        traceback.print_exc()
        return False

def test_medical_domain():
    """Test medical diagnosis domain."""
    print("\nTesting Medical Domain...")
    
    try:
        config = BirdConfig(max_factors=4)
        bird = BirdFramework(config)
        medical_domain = DomainFactory.create_domain("medical", bird)
        
        # Create test patient data
        patient = PatientData(
            age=65,
            gender="male",
            symptoms=["chest pain", "shortness of breath", "fatigue"],
            medical_history=["hypertension", "diabetes"],
            vital_signs={"blood_pressure": 160, "heart_rate": 95, "temperature": 98.6},
            lab_results={"troponin": 0.8, "creatinine": 1.2}
        )
        
        condition = "Patient reports severe chest pain that started 2 hours ago and is getting worse"
        
        print(f"Patient: {patient.age}-year-old {patient.gender}")
        print(f"Symptoms: {', '.join(patient.symptoms)}")
        print(f"Condition: {condition}")
        
        result = medical_domain.make_domain_decision(patient, condition)
        interpretation = medical_domain.interpret_result(result)
        
        print(f"\nRecommendation: {result.get_recommended_outcome()}")
        print(f"Medical confidence: {result.confidence:.3f}")
        print(f"\nInterpretation:\n{interpretation[:300]}...")
        
        # Validate medical domain
        assert result.get_recommended_outcome() in ["conservative treatment", "aggressive treatment"]
        assert medical_domain.validate_condition(condition)
        assert not medical_domain.validate_condition("invalid condition xyz")
        
        print("✓ Medical domain test passed")
        return True
        
    except Exception as e:
        print(f"✗ Medical domain test failed: {e}")
        traceback.print_exc()
        return False

def test_financial_domain():
    """Test financial investment domain."""
    print("\nTesting Financial Domain...")
    
    try:
        config = BirdConfig(max_factors=4)
        bird = BirdFramework(config)
        financial_domain = DomainFactory.create_domain("financial", bird)
        
        # Create test investment profile
        investor = InvestmentProfile(
            age=30,
            income=80000,
            net_worth=100000,
            risk_tolerance="aggressive",
            investment_goals=["retirement", "wealth_building"],
            time_horizon=30,
            current_portfolio={"stocks": 70, "bonds": 20, "cash": 10},
            market_conditions="bull market with rising interest rates"
        )
        
        condition = "Market is experiencing high volatility and inflation is rising rapidly"
        
        print(f"Investor: {investor.age} years old, ${investor.income:,} income")
        print(f"Risk tolerance: {investor.risk_tolerance}")
        print(f"Condition: {condition}")
        
        result = financial_domain.make_domain_decision(investor, condition)
        interpretation = financial_domain.interpret_result(result)
        
        print(f"\nRecommendation: {result.get_recommended_outcome()}")
        print(f"Investment confidence: {result.confidence:.3f}")
        print(f"\nInterpretation:\n{interpretation[:300]}...")
        
        # Validate financial domain
        assert result.get_recommended_outcome() in ["growth-focused strategy", "income-focused strategy"]
        assert financial_domain.validate_condition(condition)
        assert not financial_domain.validate_condition("random text")
        
        print("✓ Financial domain test passed")
        return True
        
    except Exception as e:
        print(f"✗ Financial domain test failed: {e}")
        traceback.print_exc()
        return False

def test_ecommerce_domain():
    """Test e-commerce recommendation domain."""
    print("\nTesting E-commerce Domain...")
    
    try:
        config = BirdConfig(max_factors=4)
        bird = BirdFramework(config)
        ecommerce_domain = DomainFactory.create_domain("ecommerce", bird)
        
        # Create test customer profile
        customer = CustomerProfile(
            age=25,
            gender="female",
            location="suburban area",
            income_bracket="middle income",
            purchase_history=["smartphone", "laptop", "headphones"],
            preferences={"electronics": "latest technology", "clothing": "trendy"},
            browsing_behavior=["tablets", "smartwatches", "wireless_earbuds"],
            seasonal_factors="holiday shopping season",
            budget_range="$100-300"
        )
        
        condition = "Customer is shopping for a birthday gift and needs it delivered within 3 days"
        
        print(f"Customer: {customer.age}-year-old {customer.gender} from {customer.location}")
        print(f"Recent purchases: {', '.join(customer.purchase_history[-3:])}")
        print(f"Condition: {condition}")
        
        result = ecommerce_domain.make_domain_decision(customer, condition)
        interpretation = ecommerce_domain.interpret_result(result)
        
        print(f"\nRecommendation: {result.get_recommended_outcome()}")
        print(f"Algorithm confidence: {result.confidence:.3f}")
        print(f"\nInterpretation:\n{interpretation[:300]}...")
        
        # Validate e-commerce domain
        assert result.get_recommended_outcome() in ["personalized recommendations", "trending recommendations"]
        assert ecommerce_domain.validate_condition(condition)
        assert not ecommerce_domain.validate_condition("xyz")
        
        print("✓ E-commerce domain test passed")
        return True
        
    except Exception as e:
        print(f"✗ E-commerce domain test failed: {e}")
        traceback.print_exc()
        return False

def test_error_handling():
    """Test error handling and edge cases."""
    print("\nTesting Error Handling...")
    
    try:
        config = BirdConfig()
        bird = BirdFramework(config)
        
        # Test invalid outcomes (not binary)
        try:
            invalid_context = DecisionContext(
                scenario="Test scenario",
                condition="Test condition",
                outcomes=["option1", "option2", "option3"]  # Should be binary
            )
            assert False, "Should have raised ValueError for non-binary outcomes"
        except ValueError:
            print("✓ Correctly rejected non-binary outcomes")
        
        # Test empty scenario
        try:
            empty_context = DecisionContext(
                scenario="",
                condition="Test condition",
                outcomes=["option1", "option2"]
            )
            result = bird.make_decision(empty_context)
            print("✓ Handled empty scenario gracefully")
        except Exception as e:
            print(f"✓ Appropriately handled empty scenario: {type(e).__name__}")
        
        # Test invalid domain type
        try:
            invalid_domain = DomainFactory.create_domain("invalid_domain", bird)
            assert False, "Should have raised ValueError for invalid domain"
        except ValueError:
            print("✓ Correctly rejected invalid domain type")
        
        # Test factor with no values
        try:
            invalid_factor = Factor(
                name="test",
                description="test factor",
                possible_values=[]  # Should have at least one value
            )
            assert False, "Should have raised ValueError for empty values"
        except ValueError:
            print("✓ Correctly rejected factor with no values")
        
        print("✓ Error handling test passed")
        return True
        
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        traceback.print_exc()
        return False

def test_performance():
    """Test performance characteristics."""
    print("\nTesting Performance...")
    
    try:
        config = BirdConfig(max_factors=3)  # Limit factors for faster testing
        bird = BirdFramework(config)
        
        # Test multiple decisions
        contexts = [
            DecisionContext(
                scenario="Choosing a restaurant for dinner",
                condition="It's raining and we want something quick",
                outcomes=["fast food", "sit-down restaurant"]
            ),
            DecisionContext(
                scenario="Selecting a movie to watch",
                condition="We have 2 hours and want something light",
                outcomes=["comedy movie", "drama movie"]
            ),
            DecisionContext(
                scenario="Picking a vacation destination",
                condition="Budget is limited and we prefer warm weather",
                outcomes=["beach destination", "mountain destination"]
            )
        ]
        
        total_time = 0
        successful_decisions = 0
        
        for i, context in enumerate(contexts, 1):
            try:
                start_time = time.time()
                result = bird.make_decision(context)
                end_time = time.time()
                
                decision_time = end_time - start_time
                total_time += decision_time
                successful_decisions += 1
                
                print(f"Decision {i}: {result.get_recommended_outcome()} "
                      f"(confidence: {result.confidence:.2f}, time: {decision_time:.2f}s)")
                
            except Exception as e:
                print(f"Decision {i} failed: {e}")
        
        avg_time = total_time / len(contexts) if contexts else 0
        success_rate = successful_decisions / len(contexts) if contexts else 0
        
        print(f"\nPerformance Summary:")
        print(f"Average decision time: {avg_time:.2f} seconds")
        print(f"Success rate: {success_rate:.1%}")
        print(f"Total time for {len(contexts)} decisions: {total_time:.2f} seconds")
        
        # Performance assertions
        assert avg_time < 30, f"Average decision time too slow: {avg_time:.2f}s"
        assert success_rate >= 0.8, f"Success rate too low: {success_rate:.1%}"
        
        print("✓ Performance test passed")
        return True
        
    except Exception as e:
        print(f"✗ Performance test failed: {e}")
        traceback.print_exc()
        return False

def run_comprehensive_test():
    """Run all tests and provide summary."""
    print("="*80)
    print("BIRD FRAMEWORK COMPREHENSIVE TEST SUITE")
    print("="*80)
    
    tests = [
        ("Core Framework", test_core_framework),
        ("Factor Generation", test_factor_generation),
        ("Condition Mapping", test_condition_mapping),
        ("Medical Domain", test_medical_domain),
        ("Financial Domain", test_financial_domain),
        ("E-commerce Domain", test_ecommerce_domain),
        ("Error Handling", test_error_handling),
        ("Performance", test_performance)
    ]
    
    results = {}
    start_time = time.time()
    
    for test_name, test_func in tests:
        print(f"\n{'-'*60}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            results[test_name] = False
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{test_name:<20}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total:.1%})")
    print(f"Total execution time: {total_time:.2f} seconds")
    
    if passed == total:
        print("\n🎉 All tests passed! The BIRD framework is working correctly.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the implementation.")
    
    return results

if __name__ == "__main__":
    # Check if OpenAI API key is available
    import os
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not found. Some tests may fail.")
        print("Please set your OpenAI API key as an environment variable.")
    
    # Run comprehensive test suite
    test_results = run_comprehensive_test()
    
    # Exit with appropriate code
    all_passed = all(test_results.values())
    sys.exit(0 if all_passed else 1)

