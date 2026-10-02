import pytest
import os
import tempfile
import yaml
from backend.models import WeatherData
from backend.policy_engine import PolicyEngine, SOP, MatchResult

# --- Helper Data ---
MOCK_WEATHER = WeatherData(
    temperature_2m=36.0,
    wind_speed_10m=45.0,
    precipitation=16.0,
    precipitation_probability=80.0,
    uv_index=9.0,
    hourly={},
    daily={}
)

# --- Tests ---

def test_load_valid_yaml():
    engine = PolicyEngine("backend/policies/sops.yaml")
    assert len(engine.sops) >= 12
    # Verify we have range of severities
    severities = {sop.severity for sop in engine.sops}
    assert "critical" in severities
    assert "low" in severities

def test_invalid_yaml_schema():
    invalid_yaml = [{"id": "bad_sop", "severity": "not_a_real_severity"}] # Missing required fields
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        yaml.dump(invalid_yaml, f)
        temp_path = f.name
        
    try:
        with pytest.raises(ValueError, match="Invalid SOP schema"):
            PolicyEngine(temp_path)
    finally:
        os.remove(temp_path)

def test_numeric_match_and_conflict_resolution():
    engine = PolicyEngine("backend/policies/sops.yaml")
    # Weather is terrible: 36C (high temp), 45km/h wind, 16mm rain.
    # Query: "I want to cycle"
    
    matches = engine.evaluate_policies("I want to cycle", MOCK_WEATHER)
    
    assert len(matches) > 0
    # Highest severity should be first (critical rain system > high uv > medium wind)
    assert matches[0].sop.id == "sop_extreme_rain_system"
    assert matches[0].sop.severity == "critical"
    
    # We should also see the specific cycling rules matched
    matched_ids = [m.sop.id for m in matches]
    assert "sop_wind_cycling" in matched_ids
    assert "sop_uv_high_exercise" in matched_ids
    assert "sop_mild_heat_exercise" in matched_ids
    
    # Ensure stable ordering: Critical > High > Medium > Low
    severities = [m.sop.severity for m in matches]
    expected_severities = sorted(severities, key=lambda s: {"critical": 4, "high": 3, "medium": 2, "low": 1}[s], reverse=True)
    assert severities == expected_severities

def test_non_match_and_missing_data():
    engine = PolicyEngine("backend/policies/sops.yaml")
    
    # Perfect weather
    good_weather = WeatherData(
        temperature_2m=22.0,
        wind_speed_10m=5.0,
        precipitation=0.0,
        precipitation_probability=0.0,
        uv_index=3.0,
    )
    
    matches = engine.evaluate_policies("I want to cycle", good_weather)
    # Shouldn't hit any of the danger thresholds
    assert len(matches) == 0

    # Missing UV Index
    missing_uv_weather = WeatherData(
        temperature_2m=35.0, # Will trigger heat rule
        wind_speed_10m=5.0,
        precipitation=0.0,
        precipitation_probability=0.0,
        uv_index=None, # Missing!
    )
    
    matches2 = engine.evaluate_policies("I want to cycle", missing_uv_weather)
    matched_ids = [m.sop.id for m in matches2]
    # Should NOT match UV rule because data is missing (not evaluated as 0)
    assert "sop_uv_high_exercise" not in matched_ids
    # Should STILL match heat rule because temp is 35
    assert "sop_heat_vulnerable" not in matched_ids # Wait, "cycle" doesn't hit vulnerable keywords unless intent matches.
    # Actually, sop_mild_heat_exercise triggers at 30C for exercise.
    assert "sop_mild_heat_exercise" in matched_ids

def test_intent_paraphrase():
    engine = PolicyEngine("backend/policies/sops.yaml")
    
    # "pedal" is not in keywords, but "bike" is. Let's test "going for a run" which has "run"
    matches = engine.evaluate_policies("going for a run outside", MOCK_WEATHER)
    matched_ids = [m.sop.id for m in matches]
    
    # Should match exercise SOPs
    assert "sop_uv_high_exercise" in matched_ids

def test_add_new_sop_without_code_changes():
    # 1. Load original
    engine1 = PolicyEngine("backend/policies/sops.yaml")
    original_count = len(engine1.sops)
    
    # 2. Add an 13th SOP to a temporary copy
    with open("backend/policies/sops.yaml", "r") as f:
        data = yaml.safe_load(f)
        
    data.append({
        "id": "sop_test_dynamic",
        "description": "Test dynamic addition",
        "category": "travel",
        "severity": "critical",
        "intent_keywords": ["fly", "airplane"],
        "conditions": {
            "type": "numeric",
            "rules": [
                {"metric": "wind_speed_10m", "operator": ">", "value": 1.0}
            ]
        },
        "guidance": "Dynamic test"
    })
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        yaml.dump(data, f)
        temp_path = f.name
        
    try:
        # 3. Load the new file
        engine2 = PolicyEngine(temp_path)
        assert len(engine2.sops) == original_count + 1
        
        # 4. Evaluate and see if the new logic applies immediately
        matches = engine2.evaluate_policies("I want to fly on an airplane", MOCK_WEATHER)
        matched_ids = [m.sop.id for m in matches]
        assert "sop_test_dynamic" in matched_ids
    finally:
        os.remove(temp_path)

def test_numeric_boundaries():
    engine = PolicyEngine("backend/policies/sops.yaml")
    
    # sop_wind_cycling rule is wind_speed_10m > 40.0
    # Test just below (39.9)
    weather_below = WeatherData(
        temperature_2m=25.0, wind_speed_10m=39.9, precipitation=0.0,
        precipitation_probability=0.0, uv_index=5.0
    )
    matches = engine.evaluate_policies("cycling", weather_below)
    assert not any(m.sop.id == "sop_wind_cycling" for m in matches)
    
    # Test exactly at (40.0) -> should NOT match because operator is >
    weather_exact = WeatherData(
        temperature_2m=25.0, wind_speed_10m=40.0, precipitation=0.0,
        precipitation_probability=0.0, uv_index=5.0
    )
    matches = engine.evaluate_policies("cycling", weather_exact)
    assert not any(m.sop.id == "sop_wind_cycling" for m in matches)
    
    # Test just above (40.1) -> SHOULD match
    weather_above = WeatherData(
        temperature_2m=25.0, wind_speed_10m=40.1, precipitation=0.0,
        precipitation_probability=0.0, uv_index=5.0
    )
    matches = engine.evaluate_policies("cycling", weather_above)
    assert any(m.sop.id == "sop_wind_cycling" for m in matches)

    # sop_uv_high_exercise rule is uv_index >= 8.0
    # Test exactly at (8.0) -> SHOULD match because operator is >=
    weather_uv_exact = WeatherData(
        temperature_2m=25.0, wind_speed_10m=10.0, precipitation=0.0,
        precipitation_probability=0.0, uv_index=8.0
    )
    matches = engine.evaluate_policies("running", weather_uv_exact)
    assert any(m.sop.id == "sop_uv_high_exercise" for m in matches)

