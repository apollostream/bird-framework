# BIRD Framework Architecture Design

## Core Components

### 1. BirdFramework (Main Class)
- Central orchestrator that coordinates all components
- Manages the three-stage process: Factor Generation → Condition Mapping → Probability Calculation
- Provides high-level interface for decision making

### 2. FactorGenerator
- **Purpose**: Implements abductive factor generation (Section 3.2)
- **Key Methods**:
  - `generate_factors(scenario, outcomes)`: Generate factors from scenario and outcomes
  - `prune_factors(factors, scenario, outcomes)`: Remove non-decisive factors
  - `_generate_factor_sentences()`: Generate sentences for each outcome
  - `_extract_factors_from_sentences()`: Summarize sentences into factors

### 3. ConditionMapper
- **Purpose**: Implements LLM entailment for condition-factor mapping (Section 3.3)
- **Key Methods**:
  - `map_condition_to_factors(condition, factors)`: Map condition to factor values
  - `_hierarchy_entailment()`: Use hierarchy prompt approach
  - `_direct_entailment()`: Use direct prompt approach

### 4. BayesianModel
- **Purpose**: Implements deductive Bayesian probabilistic modeling (Section 3.4)
- **Key Methods**:
  - `train(training_data)`: Train the model on factor-outcome mappings
  - `predict_probabilities(factor_values)`: Compute P(Oi|f) for given factor values
  - `_compute_marginal_probability()`: Implement marginalization over complete information space

### 5. LLMInterface
- **Purpose**: Abstraction layer for LLM interactions
- **Key Methods**:
  - `generate_text(prompt, max_tokens)`: Generate text from prompt
  - `classify(text, classes)`: Binary/multi-class classification
  - `entailment(premise, hypothesis)`: Check entailment relationship

### 6. DecisionContext
- **Purpose**: Data structure to hold decision-making context
- **Attributes**:
  - `scenario`: General situation description
  - `condition`: Additional constraints/preferences
  - `outcomes`: List of possible outcomes
  - `factors`: Generated factors and their possible values

## Data Structures

### Factor
```python
@dataclass
class Factor:
    name: str
    description: str
    possible_values: List[str]
    importance_score: float = 0.0
```

### FactorValue
```python
@dataclass
class FactorValue:
    factor_name: str
    value: str
    probability: float = 0.0
```

### DecisionResult
```python
@dataclass
class DecisionResult:
    outcome_probabilities: Dict[str, float]
    factor_mappings: List[FactorValue]
    explanation: str
    confidence: float
```

## Workflow Architecture

### Stage 1: Factor Generation
1. Input: Scenario S, Outcomes [O1, O2]
2. Generate sentences supporting each outcome
3. Extract factors from sentences
4. Prune non-decisive factors
5. Output: Factor set F with possible values

### Stage 2: Condition Mapping
1. Input: Additional condition U, Factor set F
2. Use LLM entailment to map U to factor values
3. Compute P(f|S,U) for each factor value combination
4. Output: Probability distribution over factor space

### Stage 3: Probability Calculation
1. Input: Factor probability distribution, trained Bayesian model
2. For each factor combination f, compute P(Oi|f)
3. Marginalize: P(Oi|S,U) = Σ_f P(Oi|f)P(f|S,U)
4. Output: Final outcome probabilities

## Domain Abstraction

### BaseDomain (Abstract Class)
- **Purpose**: Template for domain-specific implementations
- **Abstract Methods**:
  - `format_scenario(domain_data)`: Convert domain data to scenario text
  - `define_outcomes()`: Define domain-specific outcomes
  - `validate_condition(condition)`: Validate domain-specific conditions
  - `interpret_result(decision_result)`: Domain-specific result interpretation

### Example Domain Implementations
1. **MedicalDiagnosisDomain**: Medical decision making
2. **FinancialInvestmentDomain**: Investment decisions
3. **ProductRecommendationDomain**: E-commerce recommendations

## Configuration and Extensibility

### BirdConfig
```python
@dataclass
class BirdConfig:
    llm_model: str = "gpt-3.5-turbo"
    max_factors: int = 10
    factor_pruning_threshold: float = 0.1
    entailment_method: str = "hierarchy"  # or "direct"
    bayesian_model_type: str = "naive_bayes"
```

### Plugin Architecture
- Support for custom LLM backends
- Pluggable Bayesian model implementations
- Custom factor generation strategies
- Domain-specific preprocessing modules

## Error Handling and Validation

### Input Validation
- Scenario and condition text validation
- Outcome format validation
- Factor consistency checks

### Robustness Features
- Fallback mechanisms for LLM failures
- Confidence scoring for predictions
- Uncertainty quantification
- Graceful degradation for incomplete data

## Performance Considerations

### Caching Strategy
- Cache factor generations for similar scenarios
- Cache LLM responses for repeated queries
- Persistent storage for trained Bayesian models

### Optimization Features
- Batch processing for multiple decisions
- Parallel factor generation
- Efficient factor space exploration
- Memory-efficient probability calculations

