import json

from src.config import LLM_MODEL, MAX_IMPLICIT_NODES, TOP_K_EMBEDDING_MATCH, EMBEDDING_SIMILARITY_THRESHOLD

NER_PROMPT = """Extract named entities from the given question.

explicit_entities: Named entities directly mentioned in the question.
enhanced_entities: Entities implied by the question but not directly named.

Return ONLY valid JSON with no additional text:
{{
  "explicit_entities": [...],
  "enhanced_entities": [...]
}}

Question: {query}

JSON:"""

IMPLICIT_NODE_PROMPT = """You are identifying implicit (intermediate) nodes in a knowledge graph needed to answer a question.

The query has been mapped to these KG nodes via named entity recognition and embedding alignment:
- Explicit nodes: {explicit_nodes}
- Enhanced nodes: {enhanced_nodes}

The one-hop information around these nodes is:
{one_hop_info}

Your task: Generate up to {max_nodes} implicit nodes — intermediate KG nodes that are NOT already listed above but are needed to bridge the gap between the query and the answer. These are nodes that represent "hops" in multi-hop reasoning.

Rules:
- Must be existing nodes from the one-hop information provided
- Must be distinct from explicit and enhanced nodes
- Prefer nodes with high connectivity (appear in multiple triples)
- Limit to {max_nodes} nodes

Return as JSON array: ["node1", "node2", ...]

JSON array:"""


def _extract_json(text):
    text = text.replace("```json", "").replace("```", "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end+1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def named_entity_recognition(query, client):
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "Extract named entities from queries precisely."},
            {"role": "user", "content": NER_PROMPT.format(query=query)},
        ],
        temperature=0,
    )
    content = response.choices[0].message.content.strip()
    data = _extract_json(content)
    if data is None:
        return [], []
    explicit = data.get("explicit_entities", [])
    enhanced = data.get("enhanced_entities", [])
    return explicit, enhanced


def embedding_alignment(entities, vector_store, top_k=TOP_K_EMBEDDING_MATCH):
    aligned = []
    for entity in entities:
        matches = vector_store.search(entity, top_k=top_k)
        for node_name, score in matches:
            if score >= EMBEDDING_SIMILARITY_THRESHOLD:
                aligned.append((node_name, score))
    seen = set()
    unique_aligned = []
    for name, score in sorted(aligned, key=lambda x: -x[1]):
        if name not in seen:
            seen.add(name)
            unique_aligned.append(name)
    return unique_aligned


def node_enhancement(query, explicit_nodes, enhanced_nodes, one_hop_info, client):
    if not one_hop_info.strip():
        return []
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "Identify implicit KG nodes for multi-hop reasoning."},
            {"role": "user", "content": IMPLICIT_NODE_PROMPT.format(
                query=query,
                explicit_nodes=json.dumps(explicit_nodes),
                enhanced_nodes=json.dumps(enhanced_nodes),
                one_hop_info=one_hop_info,
                max_nodes=MAX_IMPLICIT_NODES,
            )},
        ],
        temperature=0,
    )
    content = response.choices[0].message.content.strip()
    text = content.replace("```json", "").replace("```", "").strip()
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        text = text[start:end+1]
    try:
        implicit = json.loads(text)
        if isinstance(implicit, list):
            return [n for n in implicit if n not in explicit_nodes and n not in enhanced_nodes]
    except json.JSONDecodeError:
        pass
    return []


def enhance_query(query, vector_store, neo4j_conn, client):
    explicit_entities, enhanced_entities = named_entity_recognition(query, client)
    explicit_nodes = embedding_alignment(explicit_entities, vector_store)
    enhanced_nodes_list = embedding_alignment(enhanced_entities, vector_store)
    all_aligned = list(set(explicit_nodes + enhanced_nodes_list))
    one_hop_triples = neo4j_conn.get_one_hop_triples(all_aligned) if all_aligned else []
    one_hop_text = "\n".join(
        f"({h}, {r}, {t})" for h, r, t in one_hop_triples
    )
    implicit_nodes = node_enhancement(query, explicit_nodes, enhanced_nodes_list, one_hop_text, client)
    return explicit_nodes, enhanced_nodes_list, implicit_nodes
