# BIRD Framework Implementation

A Python implementation of the **BIRD (Bayesian Inference from Abduction and Deduction)** framework for trustworthy decision-making with Large Language Models.

## Overview

This implementation is based on the paper "BIRD: A Trustworthy Bayesian Inference Framework for Large Language Models" by Yu Feng, Ben Zhou, Weidong Lin, and Dan Roth from the University of Pennsylvania.

The BIRD framework provides controllable and interpretable probability estimation for decision-making by using:
1. **Abduction**: Converting input queries into intermediate factors
2. **Deduction**: Using Bayesian modeling to estimate reliable outcome probabilities

## Key Features

- ✅ **Complete BIRD Framework Implementation**: All three stages (Factor Generation, Condition Mapping, Probability Calculation)
- ✅ **Domain-Agnostic Design**: Works across different decision-making domains
- ✅ **Three Example Domains**: Medical diagnosis, financial investment, and e-commerce recommendations
- ✅ **Configurable Parameters**: Customizable factor generation, pruning thresholds, and entailment methods
- ✅ **Robust Error Handling**: Graceful degradation and fallback mechanisms
- ✅ **Extensible Architecture**: Easy to add new domains and LLM backends

## Architecture

### Core Components

1. **BirdFramework**: Main orchestrator coordinating all components
2. **FactorGenerator**: Implements abductive factor generation
3. **ConditionMapper**: Maps conditions to factor values using LLM entailment
4. **BayesianModel**: Deductive probabilistic modeling
5. **LLMInterface**: Abstraction layer for LLM interactions

### Three-Stage Process

```
Input: Scenario + Condition + Outcomes
    ↓
Stage 1: Factor Generation (Abduction)
    ↓
Stage 2: Condition Mapping (Entailment)
    ↓
Stage 3: Probability Calculation (Deduction)
    ↓
Output: Outcome Probabilities + Explanation
```

## Installation

1. **Install Dependencies**:
```bash
pip install openai numpy
```

2. **Set up API Key**:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

3. **Download Files**:
- `bird_framework.py` - Core framework implementation
- `domain_examples.py` - Domain-specific implementations
- `demo_bird_framework.py` - Demonstration script

## Quick Start

### Basic Usage

```python
from bird_framework import BirdFramework, BirdConfig, DecisionContext

# Initialize framework
config = BirdConfig(llm_model="gpt-4.1-mini")
bird = BirdFramework(config)

# Create decision context
context = DecisionContext(
    scenario="You want to charge your phone while using it",
    condition="You will walk around the room frequently",
    outcomes=["use a shorter cord", "use a longer cord"]
)

# Make decision
result = bird.make_decision(context)

print(f"Recommendation: {result.get_recommended_outcome()}")
print(f"Confidence: {result.confidence:.3f}")
print(f"Explanation: {result.explanation}")
```

### Domain-Specific Usage

```python
from domain_examples import DomainFactory, PatientData

# Medical Domain Example
bird = BirdFramework(BirdConfig(llm_model="gpt-4.1-mini"))
medical_domain = DomainFactory.create_domain("medical", bird)

patient = PatientData(
    age=45,
    gender="female",
    symptoms=["headache", "fatigue"],
    medical_history=["hypertension"],
    vital_signs={"blood_pressure": 140, "heart_rate": 85}
)

condition = "Patient reports severe symptoms worsening over 24 hours"
result = medical_domain.make_domain_decision(patient, condition)
interpretation = medical_domain.interpret_result(result)

print(interpretation)
```

## Configuration Options

```python
config = BirdConfig(
    llm_model="gpt-4.1-mini",           # LLM model to use
    max_factors=10,                     # Maximum factors to generate
    factor_pruning_threshold=0.1,       # Threshold for factor importance
    entailment_method="hierarchy",      # "hierarchy" or "direct"
    temperature=0.7,                    # LLM temperature
    cache_enabled=True                  # Enable response caching
)
```

## Supported Domains

### 1. Medical Diagnosis Domain
- **Purpose**: Clinical decision-making support
- **Outcomes**: Conservative vs. aggressive treatment
- **Input**: Patient data, symptoms, medical history
- **Example**: Treatment approach for chest pain patient

### 2. Financial Investment Domain
- **Purpose**: Investment strategy recommendations
- **Outcomes**: Growth-focused vs. income-focused strategy
- **Input**: Investor profile, risk tolerance, market conditions
- **Example**: Portfolio allocation during market volatility

### 3. E-commerce Recommendation Domain
- **Purpose**: Product recommendation strategies
- **Outcomes**: Personalized vs. trending recommendations
- **Input**: Customer profile, purchase history, browsing behavior
- **Example**: Recommendation approach for gift shopping

## File Structure

```
bird-framework/
├── bird_framework.py          # Core framework implementation
├── domain_examples.py         # Three domain implementations
├── demo_bird_framework.py     # Demonstration script
├── test_bird_framework.py     # Comprehensive test suite
├── bird_framework_analysis.md # Paper analysis and methodology
├── bird_architecture_design.md # Architecture documentation
└── README.md                  # This file
```

## Key Classes and Methods

### BirdFramework
- `make_decision(context)` - Main decision-making method
- `train_model(training_data)` - Train the Bayesian model
- `save_framework(directory)` - Save framework state
- `load_framework(directory)` - Load framework state

### DecisionContext
- `scenario` - General situation description
- `condition` - Additional constraints/preferences
- `outcomes` - List of possible outcomes (binary)
- `factors` - Generated factors (optional)

### DecisionResult
- `outcome_probabilities` - Probability for each outcome
- `factor_mappings` - Factor values and confidences
- `explanation` - Human-readable explanation
- `confidence` - Overall confidence score
- `get_recommended_outcome()` - Highest probability outcome

## Running Examples

### 1. Basic Demonstration
```bash
python demo_bird_framework.py
```

### 2. Domain Examples
```python
from domain_examples import run_domain_examples
results = run_domain_examples()
```

### 3. Comprehensive Testing
```bash
python test_bird_framework.py
```

## Extending the Framework

### Adding a New Domain

1. **Create Domain Class**:
```python
class MyDomain(BaseDomain):
    def format_scenario(self, domain_data):
        # Convert domain data to scenario text
        pass
    
    def define_outcomes(self):
        # Return list of possible outcomes
        return ["outcome1", "outcome2"]
    
    def validate_condition(self, condition):
        # Validate domain-specific conditions
        pass
    
    def interpret_result(self, decision_result):
        # Provide domain-specific interpretation
        pass
```

2. **Register with Factory**:
```python
DomainFactory.domain_map["my_domain"] = MyDomain
```

### Custom LLM Backend

Extend `LLMInterface` to support different LLM providers:

```python
class CustomLLMInterface(LLMInterface):
    def generate_text(self, prompt, max_tokens=None):
        # Implement custom LLM call
        pass
```

## Performance Characteristics

- **Average Decision Time**: 10-30 seconds (depending on factors)
- **Factor Generation**: 2-10 factors per scenario
- **Memory Usage**: Minimal (caching optional)
- **Scalability**: Supports batch processing

## Limitations and Considerations

1. **Binary Decisions Only**: Currently supports only binary outcomes
2. **LLM Dependency**: Requires access to language models
3. **Factor Independence**: Assumes conditional independence of factors
4. **Training Data**: Bayesian model benefits from domain-specific training

## Research Paper Reference

```bibtex
@article{feng2024bird,
  title={BIRD: A Trustworthy Bayesian Inference Framework for Large Language Models},
  author={Feng, Yu and Zhou, Ben and Lin, Weidong and Roth, Dan},
  journal={arXiv preprint arXiv:2404.12494},
  year={2024}
}
```

## License

This implementation is provided for research and educational purposes. Please refer to the original paper for academic citations.

## Contributing

To contribute to this implementation:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Support

For questions or issues:
1. Check the documentation and examples
2. Review the test suite for usage patterns
3. Refer to the original paper for theoretical background

---

**Note**: This implementation demonstrates the BIRD framework concepts and provides a foundation for further research and development in trustworthy AI decision-making systems.

