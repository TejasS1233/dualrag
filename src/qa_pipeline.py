from src.config import LLM_MODEL

QA_PROMPT = """Answer the question based on the provided knowledge graph information.

The information is organized in two forms:

1. TRIPLES (local context) — (head, relation, tail) format:
{triples_text}

2. SHORTEST PATHS (global connectivity) — chains connecting multiple nodes:
{paths_text}

Use these structured facts to reason step by step about the question. The direction of relations matters.
Answer concisely and precisely.

Question: {query}

Answer:"""


def generate_answer(query, one_hop_triples, shortest_paths, client):
    triples_text = "\n".join(
        f"({h}, {r}, {t})" for h, r, t in one_hop_triples
    ) if one_hop_triples else "No triples found."

    paths_text = "\n".join(shortest_paths) if shortest_paths else "No paths found."

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "Answer questions using the provided knowledge graph facts precisely."},
            {"role": "user", "content": QA_PROMPT.format(
                triples_text=triples_text,
                paths_text=paths_text,
                query=query,
            )},
        ],
        temperature=0,
    )
    return response.choices[0].message.content.strip()
