import json, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from openai import OpenAI
from src.config import GROQ_API_KEY, GROQ_BASE_URL, LLM_MODEL

client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)

prompt = """Extract named entities from the given question.

explicit_entities: Named entities directly mentioned in the question.
enhanced_entities: Entities implied by the question but not directly named.

Return as JSON:
{
  "explicit_entities": [...],
  "enhanced_entities": [...]
}

Question: What city is the birthplace of the person who discovered radium?

JSON:"""

resp = client.chat.completions.create(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0)
print(repr(resp.choices[0].message.content))
