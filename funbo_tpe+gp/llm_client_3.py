import re
from groq import Groq

API_KEYS = [
    "xxxxxxxxxxxxxxxxxxxxxxxxx"
]

EVO_PROMPT = """
You are a Bayesian Optimization expert.
Your task is to evolve a highly optimized acquisition function for the Gaussian Process (GP) phase of a hybrid BO pipeline.

The function signature MUST be:
    def acquisition(means, variances, incumbent):

means      -> GP posterior mean
variances  -> GP posterior variance (NOT std)
incumbent  -> current best observed value

Return ONLY pure Python code. Do not include markdown formatting, explanations, or testing blocks.

[VERSION 0 (Score: {s0:.4f})]
{v0_code}

[VERSION 1 (Score: {s1:.4f})]
{v1_code}

Produce a superior VERSION 2.
"""

def generate_af(v0_code, s0, v1_code, s1):
    for key in API_KEYS:
        try:
            client = Groq(api_key=key)
            completion = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[{
                    "role": "user",
                    "content": EVO_PROMPT.format(
                        v0_code=v0_code,
                        s0=s0,
                        v1_code=v1_code,
                        s1=s1
                    )
                }],
                temperature=0.8,
                reasoning_efforts = "none",
            )
            code = completion.choices[0].message.content
            return re.sub(r"```python|```", "", code).strip()
        except Exception:
            continue
    return None