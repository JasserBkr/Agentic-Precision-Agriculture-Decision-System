"""
Week 5-6 deliverable: the system prompt(s) defining how the agent
reasons over the forecast bundle + raw signals, when it decides to call
a follow-up tool vs. finalize, and how it should write its explanation
(see the recommendation card format discussed with Claude — evidence
list + plain-language explanation, tied to reasoning_trace).

Not yet written.
"""

AGENT_SYSTEM_PROMPT = """\
TODO: write this once agent/graph.py exists. Should instruct the model to:
- Reason over forecast_bundle, ndvi_evi_ndwi, and weather_forecast together
- Only call a tool if it detects a genuine conflict or gap in evidence
- Always produce a final recommendation with an explicit evidence list and
  a short plain-language explanation, grounded in reasoning_trace
"""
