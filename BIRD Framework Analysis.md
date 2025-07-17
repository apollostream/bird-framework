# BIRD Framework Analysis

## Paper: BIRD: A Trustworthy Bayesian Inference Framework for Large Language Models

### Key Concepts:

1. **Problem**: LLMs rely on inductive reasoning for decision making, which results in unreliable decisions when applied to real-world tasks with incomplete contexts and conditions.

2. **Solution**: BIRD (Bayesian Inference from Abduction and Deduction) framework that provides controllable and interpretable probability estimation for model decisions.

### Framework Components:

1. **Abduction**: X → Z (conceptualize input query into intermediate factors)
2. **Deduction**: X,Z → Y (fit Bayesian model based on factors to estimate outcome probabilities)

### Problem Formulation:
- Given context C = (S, U) where:
  - S = scenario (general situation)
  - U = additional condition (constraints/preferences)
- Task: Make binary decision between outcomes O1 and O2
- Goal: Compute P(Oi|C) for i=1,2

### Three-Stage Process:
1. **Factor Generation**: Use LLMs to abductively conceptualize S and O to a set of factors (similar to Bayesian networks) to build complete information space F
2. **Condition Mapping**: Use LLM entailment to map (S,U) to factors to compute P(F|S,U)
3. **Probability Calculation**: Learn to compute P(Oi|F) using learnable text-based Bayesian model, then estimate P(Oi|S,U) through P(Oi|F) and P(F|S,U)

### Key Properties:
- **Interpretable**: Intermediate symbolic factor structure illustrates deductive reasoning process
- **Controllable**: Consistent mapping to same factor structure, probability estimation based solely on factors, allows human preference injection
- **Reliable**: Produces probability estimations that align with human judgments over 65% of the time



## Detailed Methodology:

### 3.2 Abductive Factor Generation
- For scenario S, use LLMs to derive N factors {Fj}^N_{j=1}
- Each factor Fj is a discrete variable with value set Fj containing all possible details
- Use product space F = ∏^N_{j=1} Fj to denote complete information space
- Two-stage approach:
  1. Generate sentences describing situations that increase likelihood of each outcome
  2. Summarize these sentences into factors with corresponding values
- Use binary classification to assess how each factor value influences decisions
- Factor pruning: eliminate factors that are not crucial for outcome prediction

### 3.3 LLM Entailment
- Given additional condition U under scenario S, identify which factors and values are implied
- Use entailment task formulation suitable for existing LLMs
- Two prompt approaches:
  1. Hierarchy prompt: first identify implied factors, then choose most implied value
  2. Direct prompt: directly ask if context entails a value from a factor
- Map all additional conditions to same factor structure for controllability

### 3.4 Deductive Bayesian Probabilistic Modeling
- Use text-based Bayesian modeling where each parameter is a phrase
- Apply Law of total probability to differentiate between world modeling and observations
- Predictive probability obtained by marginalizing over complete information space:
  P(Oi | C) = Σ_{f∈F} P(Oi | f)P(f | C)
- Where i = 1,2, C = (S,U), f = (f1, f2, ..., fN), fj ∈ Fj
- Assumption: factors are conditionally independent given the scenario

### Key Mathematical Formulation:
- Context C = (S, U) where S = scenario, U = additional condition
- Binary decision between outcomes O1 and O2
- Complete information space F built from factors
- Final probability: P(Oi|S,U) through P(Oi|F) and P(F|S,U)

### Implementation Steps:
1. **Factor Generation**: Use LLM to generate factors from scenario and outcomes
2. **Factor Pruning**: Use binary classification to keep only decisive factors
3. **Condition Mapping**: Use LLM entailment to map conditions to factor values
4. **Probability Calculation**: Use Bayesian model to compute final probabilities

