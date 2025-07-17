"""
BIRD: A Trustworthy Bayesian Inference Framework for Large Language Models

This module implements the BIRD framework as described in the paper:
"BIRD: A Trustworthy Bayesian Inference Framework for Large Language Models"
by Yu Feng, Ben Zhou, Weidong Lin, Dan Roth (University of Pennsylvania)

The framework provides controllable and interpretable probability estimation
for decision-making using abduction and deduction with LLMs.
"""

import json
import logging
import numpy as np
import openai
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import defaultdict
import itertools
import pickle
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Factor:
    """Represents a factor in the Bayesian network."""
    name: str
    description: str
    possible_values: List[str]
    importance_score: float = 0.0
    
    def __post_init__(self):
        if not self.possible_values:
            raise ValueError(f"Factor {self.name} must have at least one possible value")

@dataclass
class FactorValue:
    """Represents a specific value assignment for a factor."""
    factor_name: str
    value: str
    probability: float = 0.0
    
    def __post_init__(self):
        if not 0 <= self.probability <= 1:
            raise ValueError(f"Probability must be between 0 and 1, got {self.probability}")

@dataclass
class DecisionContext:
    """Holds the context for a decision-making problem."""
    scenario: str
    condition: str
    outcomes: List[str]
    factors: List[Factor] = field(default_factory=list)
    
    def __post_init__(self):
        if len(self.outcomes) != 2:
            raise ValueError("BIRD framework currently supports binary decisions only")

@dataclass
class DecisionResult:
    """Result of the BIRD decision-making process."""
    outcome_probabilities: Dict[str, float]
    factor_mappings: List[FactorValue]
    explanation: str
    confidence: float
    
    def get_recommended_outcome(self) -> str:
        """Returns the outcome with highest probability."""
        return max(self.outcome_probabilities.items(), key=lambda x: x[1])[0]

@dataclass
class BirdConfig:
    """Configuration for the BIRD framework."""
    llm_model: str = "gpt-3.5-turbo"
    max_factors: int = 10
    factor_pruning_threshold: float = 0.1
    entailment_method: str = "hierarchy"  # or "direct"
    bayesian_model_type: str = "naive_bayes"
    temperature: float = 0.7
    max_tokens: int = 1000
    cache_enabled: bool = True

class LLMInterface:
    """Interface for interacting with Large Language Models."""
    
    def __init__(self, config: BirdConfig):
        self.config = config
        self.client = openai.OpenAI()
        self._cache = {} if config.cache_enabled else None
    
    def generate_text(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """Generate text using the LLM."""
        cache_key = hash(prompt) if self._cache is not None else None
        
        if cache_key and cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens or self.config.max_tokens,
                temperature=self.config.temperature
            )
            result = response.choices[0].message.content.strip()
            
            if cache_key:
                self._cache[cache_key] = result
            
            return result
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise
    
    def classify(self, text: str, classes: List[str]) -> str:
        """Perform classification using the LLM."""
        prompt = f"""
        Classify the following text into one of these categories: {', '.join(classes)}
        
        Text: {text}
        
        Respond with only the category name.
        """
        return self.generate_text(prompt, max_tokens=50)
    
    def entailment(self, premise: str, hypothesis: str) -> float:
        """Check entailment relationship and return confidence score."""
        prompt = f"""
        Does the premise entail the hypothesis? Respond with a confidence score between 0 and 1.
        
        Premise: {premise}
        Hypothesis: {hypothesis}
        
        Respond with only a number between 0 and 1.
        """
        try:
            response = self.generate_text(prompt, max_tokens=10)
            score = float(re.search(r'0?\.\d+|[01]', response).group())
            return max(0.0, min(1.0, score))
        except:
            return 0.5  # Default to neutral if parsing fails

class FactorGenerator:
    """Generates factors from scenarios and outcomes using abductive reasoning."""
    
    def __init__(self, llm_interface: LLMInterface, config: BirdConfig):
        self.llm = llm_interface
        self.config = config
    
    def generate_factors(self, scenario: str, outcomes: List[str]) -> List[Factor]:
        """Generate factors from scenario and outcomes."""
        logger.info(f"Generating factors for scenario: {scenario[:50]}...")
        
        # Stage 1: Generate sentences for each outcome
        outcome_sentences = {}
        for outcome in outcomes:
            sentences = self._generate_factor_sentences(scenario, outcome)
            outcome_sentences[outcome] = sentences
        
        # Stage 2: Extract factors from sentences
        factors = self._extract_factors_from_sentences(outcome_sentences)
        
        # Stage 3: Prune non-decisive factors
        pruned_factors = self._prune_factors(factors, scenario, outcomes)
        
        logger.info(f"Generated {len(pruned_factors)} factors after pruning")
        return pruned_factors
    
    def _generate_factor_sentences(self, scenario: str, outcome: str) -> List[str]:
        """Generate sentences that support a specific outcome."""
        prompt = f"""
        Given the scenario: "{scenario}"
        
        Generate 5-7 specific situations or conditions that would increase the likelihood of choosing: "{outcome}"
        
        Each situation should be a complete sentence describing a specific factor that influences the decision.
        Format as a numbered list.
        """
        
        response = self.llm.generate_text(prompt)
        sentences = []
        
        for line in response.split('\n'):
            line = line.strip()
            if re.match(r'^\d+\.', line):
                sentence = re.sub(r'^\d+\.\s*', '', line)
                if sentence:
                    sentences.append(sentence)
        
        return sentences[:7]  # Limit to 7 sentences
    
    def _extract_factors_from_sentences(self, outcome_sentences: Dict[str, List[str]]) -> List[Factor]:
        """Extract factors from generated sentences."""
        all_sentences = []
        for sentences in outcome_sentences.values():
            all_sentences.extend(sentences)
        
        prompt = f"""
        Analyze the following sentences and extract the key decision factors. Each factor should be:
        1. A general concept that can have multiple possible values
        2. Relevant to the decision-making process
        3. Independent from other factors
        
        Sentences:
        {chr(10).join(f"- {s}" for s in all_sentences)}
        
        For each factor, provide:
        - Factor name (short, descriptive)
        - Factor description (one sentence)
        - Possible values (2-4 options)
        
        Format as JSON:
        [
            {{
                "name": "factor_name",
                "description": "factor description",
                "possible_values": ["value1", "value2", "value3"]
            }}
        ]
        """
        
        response = self.llm.generate_text(prompt)
        
        try:
            # Extract JSON from response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                factors_data = json.loads(json_match.group())
                factors = []
                
                for factor_data in factors_data[:self.config.max_factors]:
                    factor = Factor(
                        name=factor_data['name'],
                        description=factor_data['description'],
                        possible_values=factor_data['possible_values']
                    )
                    factors.append(factor)
                
                return factors
        except Exception as e:
            logger.error(f"Failed to parse factors: {e}")
        
        # Fallback: create generic factors
        return self._create_fallback_factors()
    
    def _create_fallback_factors(self) -> List[Factor]:
        """Create fallback factors when extraction fails."""
        return [
            Factor(
                name="urgency",
                description="How urgent the decision is",
                possible_values=["low", "medium", "high"]
            ),
            Factor(
                name="cost",
                description="Cost implications of the decision",
                possible_values=["low", "medium", "high"]
            ),
            Factor(
                name="convenience",
                description="How convenient the option is",
                possible_values=["inconvenient", "neutral", "convenient"]
            )
        ]
    
    def _prune_factors(self, factors: List[Factor], scenario: str, outcomes: List[str]) -> List[Factor]:
        """Prune factors that are not decisive for the outcomes."""
        decisive_factors = []
        
        for factor in factors:
            importance = self._assess_factor_importance(factor, scenario, outcomes)
            if importance >= self.config.factor_pruning_threshold:
                factor.importance_score = importance
                decisive_factors.append(factor)
        
        return decisive_factors
    
    def _assess_factor_importance(self, factor: Factor, scenario: str, outcomes: List[str]) -> float:
        """Assess how important a factor is for decision making."""
        prompt = f"""
        Given the scenario: "{scenario}"
        And the decision between: {' vs '.join(outcomes)}
        
        How important is the factor "{factor.name}" ({factor.description}) for making this decision?
        
        Consider:
        - Does this factor significantly influence the choice between outcomes?
        - Would different values of this factor lead to different decisions?
        
        Respond with a score between 0 and 1, where:
        - 0 = Not important at all
        - 0.5 = Moderately important
        - 1 = Extremely important
        
        Respond with only the number.
        """
        
        try:
            response = self.llm.generate_text(prompt, max_tokens=10)
            score = float(re.search(r'0?\.\d+|[01]', response).group())
            return max(0.0, min(1.0, score))
        except:
            return 0.5  # Default to moderate importance

class ConditionMapper:
    """Maps conditions to factor values using LLM entailment."""
    
    def __init__(self, llm_interface: LLMInterface, config: BirdConfig):
        self.llm = llm_interface
        self.config = config
    
    def map_condition_to_factors(self, condition: str, factors: List[Factor]) -> List[FactorValue]:
        """Map a condition to factor values using entailment."""
        logger.info(f"Mapping condition to factors: {condition[:50]}...")
        
        factor_mappings = []
        
        for factor in factors:
            if self.config.entailment_method == "hierarchy":
                factor_value = self._hierarchy_entailment(condition, factor)
            else:
                factor_value = self._direct_entailment(condition, factor)
            
            if factor_value:
                factor_mappings.append(factor_value)
        
        return factor_mappings
    
    def _hierarchy_entailment(self, condition: str, factor: Factor) -> Optional[FactorValue]:
        """Use hierarchy approach: first check if factor is implied, then find value."""
        # Step 1: Check if the factor is implied by the condition
        factor_relevance_prompt = f"""
        Given the condition: "{condition}"
        
        Is the factor "{factor.name}" ({factor.description}) relevant or implied by this condition?
        
        Respond with YES or NO.
        """
        
        relevance_response = self.llm.generate_text(factor_relevance_prompt, max_tokens=10)
        
        if "YES" not in relevance_response.upper():
            return None
        
        # Step 2: Find the most implied value
        value_scores = {}
        for value in factor.possible_values:
            score = self._compute_entailment_score(condition, factor, value)
            value_scores[value] = score
        
        # Return the value with highest entailment score
        best_value = max(value_scores.items(), key=lambda x: x[1])
        
        if best_value[1] > 0.5:  # Only return if confidence is above threshold
            return FactorValue(
                factor_name=factor.name,
                value=best_value[0],
                probability=best_value[1]
            )
        
        return None
    
    def _direct_entailment(self, condition: str, factor: Factor) -> Optional[FactorValue]:
        """Use direct approach: directly check entailment for each value."""
        value_scores = {}
        
        for value in factor.possible_values:
            score = self._compute_entailment_score(condition, factor, value)
            value_scores[value] = score
        
        # Return the value with highest entailment score
        best_value = max(value_scores.items(), key=lambda x: x[1])
        
        if best_value[1] > 0.3:  # Lower threshold for direct method
            return FactorValue(
                factor_name=factor.name,
                value=best_value[0],
                probability=best_value[1]
            )
        
        return None
    
    def _compute_entailment_score(self, condition: str, factor: Factor, value: str) -> float:
        """Compute entailment score between condition and factor value."""
        hypothesis = f"The {factor.name} is {value}"
        return self.llm.entailment(condition, hypothesis)

class BayesianModel:
    """Implements the deductive Bayesian probabilistic modeling."""
    
    def __init__(self, config: BirdConfig):
        self.config = config
        self.factor_outcome_probs = {}  # P(Oi | factor_values)
        self.is_trained = False
    
    def train(self, training_data: List[Tuple[List[FactorValue], str]]):
        """Train the Bayesian model on factor-outcome mappings."""
        logger.info(f"Training Bayesian model with {len(training_data)} examples")
        
        # Count occurrences for naive Bayes
        outcome_counts = defaultdict(int)
        factor_value_counts = defaultdict(lambda: defaultdict(int))
        factor_value_outcome_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        
        for factor_values, outcome in training_data:
            outcome_counts[outcome] += 1
            
            for fv in factor_values:
                factor_value_counts[fv.factor_name][fv.value] += 1
                factor_value_outcome_counts[fv.factor_name][fv.value][outcome] += 1
        
        # Compute probabilities with Laplace smoothing
        total_examples = len(training_data)
        
        for factor_name in factor_value_counts:
            for value in factor_value_counts[factor_name]:
                for outcome in outcome_counts:
                    count = factor_value_outcome_counts[factor_name][value][outcome]
                    total_for_value = factor_value_counts[factor_name][value]
                    
                    # Laplace smoothing
                    prob = (count + 1) / (total_for_value + len(outcome_counts))
                    
                    key = (factor_name, value, outcome)
                    self.factor_outcome_probs[key] = prob
        
        self.is_trained = True
    
    def predict_probabilities(self, factor_values: List[FactorValue], outcomes: List[str]) -> Dict[str, float]:
        """Compute P(Oi|factor_values) for each outcome."""
        if not self.is_trained:
            # Use uniform distribution if not trained
            return {outcome: 1.0 / len(outcomes) for outcome in outcomes}
        
        outcome_probs = {}
        
        for outcome in outcomes:
            prob = 1.0
            
            for fv in factor_values:
                key = (fv.factor_name, fv.value, outcome)
                factor_prob = self.factor_outcome_probs.get(key, 0.5)  # Default to neutral
                prob *= factor_prob
            
            outcome_probs[outcome] = prob
        
        # Normalize probabilities
        total_prob = sum(outcome_probs.values())
        if total_prob > 0:
            outcome_probs = {k: v / total_prob for k, v in outcome_probs.items()}
        else:
            outcome_probs = {outcome: 1.0 / len(outcomes) for outcome in outcomes}
        
        return outcome_probs
    
    def save_model(self, filepath: str):
        """Save the trained model to disk."""
        model_data = {
            'factor_outcome_probs': dict(self.factor_outcome_probs),
            'is_trained': self.is_trained,
            'config': self.config
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """Load a trained model from disk."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.factor_outcome_probs = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.factor_outcome_probs.update(model_data['factor_outcome_probs'])
        self.is_trained = model_data['is_trained']

class BirdFramework:
    """Main BIRD framework orchestrator."""
    
    def __init__(self, config: Optional[BirdConfig] = None):
        self.config = config or BirdConfig()
        self.llm = LLMInterface(self.config)
        self.factor_generator = FactorGenerator(self.llm, self.config)
        self.condition_mapper = ConditionMapper(self.llm, self.config)
        self.bayesian_model = BayesianModel(self.config)
    
    def make_decision(self, context: DecisionContext) -> DecisionResult:
        """Make a decision using the BIRD framework."""
        logger.info(f"Making decision for scenario: {context.scenario[:50]}...")
        
        # Stage 1: Generate factors if not provided
        if not context.factors:
            context.factors = self.factor_generator.generate_factors(
                context.scenario, context.outcomes
            )
        
        # Stage 2: Map condition to factor values
        factor_mappings = self.condition_mapper.map_condition_to_factors(
            context.condition, context.factors
        )
        
        # Stage 3: Compute outcome probabilities
        outcome_probs = self.bayesian_model.predict_probabilities(
            factor_mappings, context.outcomes
        )
        
        # Generate explanation
        explanation = self._generate_explanation(context, factor_mappings, outcome_probs)
        
        # Compute confidence score
        confidence = self._compute_confidence(outcome_probs, factor_mappings)
        
        return DecisionResult(
            outcome_probabilities=outcome_probs,
            factor_mappings=factor_mappings,
            explanation=explanation,
            confidence=confidence
        )
    
    def train_model(self, training_data: List[Tuple[DecisionContext, str]]):
        """Train the Bayesian model with labeled examples."""
        processed_data = []
        
        for context, outcome in training_data:
            # Generate factors for this context
            if not context.factors:
                context.factors = self.factor_generator.generate_factors(
                    context.scenario, context.outcomes
                )
            
            # Map condition to factors
            factor_mappings = self.condition_mapper.map_condition_to_factors(
                context.condition, context.factors
            )
            
            processed_data.append((factor_mappings, outcome))
        
        self.bayesian_model.train(processed_data)
    
    def _generate_explanation(self, context: DecisionContext, 
                            factor_mappings: List[FactorValue], 
                            outcome_probs: Dict[str, float]) -> str:
        """Generate human-readable explanation for the decision."""
        recommended_outcome = max(outcome_probs.items(), key=lambda x: x[1])[0]
        confidence_pct = outcome_probs[recommended_outcome] * 100
        
        explanation = f"Based on the analysis, I recommend '{recommended_outcome}' with {confidence_pct:.1f}% confidence.\n\n"
        
        if factor_mappings:
            explanation += "Key factors considered:\n"
            for fm in factor_mappings:
                explanation += f"- {fm.factor_name}: {fm.value} (confidence: {fm.probability:.2f})\n"
        else:
            explanation += "No specific factors were strongly indicated by the given condition.\n"
        
        explanation += f"\nOutcome probabilities:\n"
        for outcome, prob in outcome_probs.items():
            explanation += f"- {outcome}: {prob:.3f}\n"
        
        return explanation
    
    def _compute_confidence(self, outcome_probs: Dict[str, float], 
                          factor_mappings: List[FactorValue]) -> float:
        """Compute overall confidence in the decision."""
        # Confidence based on probability spread and factor certainty
        prob_values = list(outcome_probs.values())
        max_prob = max(prob_values)
        prob_spread = max_prob - min(prob_values)
        
        # Factor certainty
        if factor_mappings:
            avg_factor_confidence = sum(fm.probability for fm in factor_mappings) / len(factor_mappings)
        else:
            avg_factor_confidence = 0.5
        
        # Combine both measures
        confidence = (prob_spread + avg_factor_confidence) / 2
        return min(1.0, confidence)
    
    def save_framework(self, directory: str):
        """Save the entire framework state."""
        Path(directory).mkdir(exist_ok=True)
        
        # Save Bayesian model
        self.bayesian_model.save_model(f"{directory}/bayesian_model.pkl")
        
        # Save configuration
        with open(f"{directory}/config.json", 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
    
    def load_framework(self, directory: str):
        """Load framework state from directory."""
        # Load Bayesian model
        model_path = f"{directory}/bayesian_model.pkl"
        if Path(model_path).exists():
            self.bayesian_model.load_model(model_path)
        
        # Load configuration
        config_path = f"{directory}/config.json"
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                config_data = json.load(f)
                self.config = BirdConfig(**config_data)

# Abstract base class for domain-specific implementations
class BaseDomain(ABC):
    """Abstract base class for domain-specific BIRD implementations."""
    
    def __init__(self, bird_framework: BirdFramework):
        self.bird = bird_framework
    
    @abstractmethod
    def format_scenario(self, domain_data: Any) -> str:
        """Convert domain-specific data to scenario text."""
        pass
    
    @abstractmethod
    def define_outcomes(self) -> List[str]:
        """Define domain-specific outcomes."""
        pass
    
    @abstractmethod
    def validate_condition(self, condition: str) -> bool:
        """Validate domain-specific conditions."""
        pass
    
    @abstractmethod
    def interpret_result(self, decision_result: DecisionResult) -> str:
        """Provide domain-specific interpretation of results."""
        pass
    
    def make_domain_decision(self, domain_data: Any, condition: str) -> DecisionResult:
        """Make a decision using domain-specific formatting."""
        scenario = self.format_scenario(domain_data)
        outcomes = self.define_outcomes()
        
        if not self.validate_condition(condition):
            raise ValueError(f"Invalid condition for this domain: {condition}")
        
        context = DecisionContext(
            scenario=scenario,
            condition=condition,
            outcomes=outcomes
        )
        
        return self.bird.make_decision(context)

if __name__ == "__main__":
    # Example usage
    config = BirdConfig(llm_model="gpt-3.5-turbo")
    bird = BirdFramework(config)
    
    # Example decision context
    context = DecisionContext(
        scenario="You want to charge your phone while using it",
        condition="You will walk around the room",
        outcomes=["use a shorter cord", "use a longer cord"]
    )
    
    result = bird.make_decision(context)
    print(f"Recommendation: {result.get_recommended_outcome()}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Explanation:\n{result.explanation}")

