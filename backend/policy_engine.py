import yaml
import operator
from enum import Enum
from typing import List, Optional, Literal, Dict, Any, Tuple
from pydantic import BaseModel, Field, ValidationError

from backend.models import WeatherData

class Operator(str, Enum):
    GT = ">"
    LT = "<"
    GTE = ">="
    LTE = "<="
    EQ = "=="
    NEQ = "!="

class NumericRule(BaseModel):
    metric: str
    operator: Operator
    value: float

class Condition(BaseModel):
    type: Literal["numeric", "fuzzy"]
    rules: Optional[List[NumericRule]] = None
    prompt: Optional[str] = None

class SOP(BaseModel):
    id: str
    description: str
    category: str
    severity: Literal["critical", "high", "medium", "low"]
    intent_keywords: List[str]
    conditions: Condition
    guidance: str

class MatchResult(BaseModel):
    sop: SOP
    evidence: str

class PolicyEngine:
    def __init__(self, yaml_path: str):
        self.yaml_path = yaml_path
        self.sops: List[SOP] = []
        self._load_sops()

    def _load_sops(self):
        try:
            with open(self.yaml_path, "r") as f:
                data = yaml.safe_load(f)
            
            if not isinstance(data, list):
                raise ValueError("YAML root must be a list of SOPs")
            
            for i, item in enumerate(data):
                try:
                    self.sops.append(SOP(**item))
                except ValidationError as e:
                    raise ValueError(f"Invalid SOP schema at index {i} (ID: {item.get('id', 'unknown')}):\n{e}")
        except FileNotFoundError:
            raise FileNotFoundError(f"SOPs file not found at {self.yaml_path}")

    def _get_metric_value(self, metric_path: str, weather_data: WeatherData) -> Optional[float]:
        """
        Extracts a metric using dot notation, e.g., 'current.temperature_2m'
        Returns None if the path doesn't exist or is missing in optional blocks.
        """
        parts = metric_path.split('.')
        
        # Base starts at weather_data model dump
        current_obj = weather_data.model_dump()
        for part in parts:
            if not isinstance(current_obj, dict) or part not in current_obj:
                return None
            current_obj = current_obj[part]
            if current_obj is None:
                return None # The field exists but is explicitly None (e.g. optional fields)
                
        return float(current_obj)

    def _evaluate_numeric_condition(self, rules: List[NumericRule], weather_data: WeatherData) -> Tuple[bool, str]:
        """
        Evaluates numeric rules deterministically. 
        Returns (is_match, evidence_string).
        If ANY rule fails (or data is missing), it does not match.
        """
        ops = {
            Operator.GT: operator.gt,
            Operator.LT: operator.lt,
            Operator.GTE: operator.ge,
            Operator.LTE: operator.le,
            Operator.EQ: operator.eq,
            Operator.NEQ: operator.ne,
        }
        
        evidence = []
        for rule in rules:
            actual_value = self._get_metric_value(rule.metric, weather_data)
            
            if actual_value is None:
                # Field is missing; we do not assume it's zero or false. We explicitly fail the match.
                return False, f"Missing data for {rule.metric}"
            
            op_func = ops[rule.operator]
            if not op_func(actual_value, rule.value):
                return False, f"{rule.metric} ({actual_value}) not {rule.operator.value} {rule.value}"
            
            evidence.append(f"{rule.metric} ({actual_value}) {rule.operator.value} {rule.value}")
            
        return True, " AND ".join(evidence)

    def evaluate_intent(self, query: str) -> List[str]:
        """
        Placeholder for semantic intent extraction.
        For Stage 3, we do simple keyword matching from the query.
        """
        query_lower = query.lower()
        matched_categories = set()
        
        # Always include 'all' category (e.g., severe weather overriding)
        matched_categories.add("all")
        
        for sop in self.sops:
            for keyword in sop.intent_keywords:
                if keyword.lower() in query_lower:
                    matched_categories.add(sop.category)
                    
        return list(matched_categories)

    def evaluate_policies(self, user_query: str, weather_data: WeatherData) -> List[MatchResult]:
        """
        Evaluates all SOPs against the user's intent and weather data.
        Returns a sorted list of matched SOPs based on severity (critical > high > medium > low) and then ID.
        """
        matched_categories = self.evaluate_intent(user_query)
        matches = []
        
        severity_rank = {
            "critical": 4,
            "high": 3,
            "medium": 2,
            "low": 1
        }
        
        for sop in self.sops:
            # 1. Filter by intent/category
            if sop.category not in matched_categories and not sop.intent_keywords:
                # If no keywords but category isn't matched and isn't 'all', check if we match it.
                # Actually, our simple intent matcher adds categories.
                if sop.category not in matched_categories:
                    continue
            elif sop.intent_keywords and not any(kw.lower() in user_query.lower() for kw in sop.intent_keywords):
                # Fallback simple check
                continue

            # 2. Evaluate Conditions
            if sop.conditions.type == "numeric":
                if not sop.conditions.rules:
                    continue
                is_match, evidence = self._evaluate_numeric_condition(sop.conditions.rules, weather_data)
                if is_match:
                    matches.append(MatchResult(sop=sop, evidence=evidence))
            elif sop.conditions.type == "fuzzy":
                # For Stage 3, fuzzy matching isn't executed against an LLM.
                # We skip or mock it.
                pass

        # 3. Conflict Resolution (Sorting)
        # Sort by severity (descending), then by ID (ascending) to ensure deterministic tie-breaking.
        matches.sort(key=lambda m: (severity_rank.get(m.sop.severity, 0), m.sop.id), reverse=True)
        # Re-reverse ID so ID is ascending while severity is descending
        matches.sort(key=lambda m: severity_rank.get(m.sop.severity, 0), reverse=True)
        # Python sort is stable. Let's do it cleanly:
        matches = sorted(matches, key=lambda m: (-severity_rank.get(m.sop.severity, 0), m.sop.id))
        
        return matches
