# Evaluation Suite Report

## Overview
The evaluation suite for the Weather-Advisory Support Bot ensures reliable, deterministic, and isolated testing of the system's core orchestration (LangGraph) and policy engine. It rigorously evaluates edge cases, prompt constraints, and policy adherence without requiring external network calls.

## What the Evaluation Suite Covers
1. **Intent and Paraphrase Handling:** Validates intent extraction against varied phrasings, indirect requests (e.g., "stretch my legs"), and cross-turn conversational context (follow-up questions).
2. **Numeric SOPs & Boundary Conditions:** Tests weather metric evaluations at exact boundaries (just below, exactly at, and just above threshold values) to ensure deterministic logic matching.
3. **Fuzzy SOPs:** Tests logic for subjective/fuzzy rules (e.g., stargazing visibility) under both positive and negative matched LLM outputs.
4. **Severe and Cross-Category Weather:** Verifies that global rules with high severity (e.g., extreme rain systems) override local activity-specific SOPs and aren't suppressed during filtering.
5. **Missing Weather Data:** Ensures missing API data (e.g., `None` for optional fields like UV index) doesn't erroneously trigger numerical SOPs or default to unsafe values (like 0).
6. **API and Model Failures:** Validates graceful fallback and robust error handling during geocoding failures, weather HTTP timeouts, malformed API schemas, and unexpected LLM exceptions, ensuring the bot remains honest and doesn't fabricate observations.
7. **No Applicable Guidance:** Verifies the system actively states "I don't have guidance for that" when no SOP applies, explicitly testing the final generated API response.
8. **Prompt Injection:** Checks that safety instructions block requests to "ignore policies," preventing users from bypassing authoritative rules or receiving fabricated weather values.
9. **Weather Grounding and SOP Traceability:** Confirms reported numbers stem strictly from API-provided arrays and recommendations accurately map to real, referenced YAML SOP identifiers.
10. **Session Memory and Isolation:** Proves proper state retention within a continuous `session_id` while strongly isolating concurrent but distinct `session_id` threads.

## Mocked Dependencies
To maintain fast, deterministic, and offline testing, the following network dependencies are fully mocked via `unittest.mock`:
* **Open-Meteo Weather API:** Async HTTP clients in `get_weather` are mocked to return static responses, HTTP error traces, or malformed JSON payloads.
* **Geocoding API:** Location-to-coordinates HTTP requests (`get_location`) are mocked.
* **Groq LLM Models:** `ChatGroq` invocation flows (`invoke` and `with_structured_output().invoke`/`ainvoke`) are stubbed out to return predefined AIMessages or Pydantic output schemas (e.g., `IntentOutput`, `FuzzyResult`).

## How to Run the Tests
You can run the full backend evaluation suite from the project root. Network access and a real `GROQ_API_KEY` are not required:

```powershell
$env:PYTHONPATH='.'
.\backend\venv\Scripts\pytest backend\
```

## Known Limitations
* **LLM Variability:** Mocking the LLM ensures tests are deterministic but inherently assumes the real LLM will consistently output the exact structured schema (like `IntentOutput`) as expected. It does not measure the actual zero-shot success rate of the model on unseen data.
* **Schema Drift:** Open-Meteo API response structures are mocked. If Open-Meteo dramatically changes its payload format in production, the test suite may incorrectly report success since it validates against stale schema fixtures.
* **Hallucinations:** While the tests verify the anti-fabrication prompt instructions are included and correctly route data, they cannot guarantee the live Groq model will absolutely never hallucinate a metric in a novel adversarial setup.

## Scenarios Requiring Manual or Live-API Verification
The following scenarios must be validated periodically against live environments:
1. **Model Degradation:** Spot-testing live chats to verify Groq's intent classification handles highly ambiguous slang that wasn't conceived during test generation.
2. **API Latency and Limits:** Real-world timeout behaviors, rate-limiting (`429 Too Many Requests`), or geolocation DNS failures which can't be purely simulated by instant mock returns.
3. **Fuzzy Prompt Realism:** Verifying that the LLM actually correctly understands subjective physical conditions, such as accurately correlating cloud cover to "stargazing visibility" in real time.
