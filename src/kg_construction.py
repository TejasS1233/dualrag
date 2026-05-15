import re
import json
from neo4j import GraphDatabase
import chromadb
from sentence_transformers import SentenceTransformer

from src.config import (
    GROQ_API_KEY, GROQ_BASE_URL, LLM_MODEL,
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    CHROMA_DB_PATH, EMBEDDING_MODEL,
    CHUNK_SIZE, CHUNK_OVERLAP,
)

TRIPLE_EXTRACTION_PROMPT = """You are a knowledge graph construction assistant. Extract all factual triples from the given text.

A triple has the format: (head, relation, tail)
- head and tail must be the most concise form of entities
- relation must be precise and specific
- Resolve all pronouns and abbreviations to their full canonical forms
- Extract parenthetical information as separate triples

Output each triple on a separate line using the format:
(head|relation|tail)

Text:
{text}

Triples:"""


def chunk_corpus(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
    return chunks


def extract_triples(chunk, client):
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You extract triples from text precisely and concisely."},
            {"role": "user", "content": TRIPLE_EXTRACTION_PROMPT.format(text=chunk)},
        ],
        temperature=0,
    )
    content = response.choices[0].message.content.strip()
    triples = []
    for line in content.split("\n"):
        line = line.strip()
        if "|" in line and line.startswith("(") and line.endswith(")"):
            parts = line[1:-1].split("|")
            if len(parts) == 3:
                h, r, t = [p.strip() for p in parts]
                if h and r and t:
                    triples.append((h, r, t))
        elif "|" in line:
            parts = line.split("|")
            if len(parts) == 3:
                h, r, t = [p.strip(" ()").strip() for p in parts]
                if h and r and t:
                    triples.append((h, r, t))
    return triples


def deduplicate_triples(triples):
    seen = set()
    unique = []
    for h, r, t in triples:
        key = (h.lower(), r.lower(), t.lower())
        if key not in seen:
            seen.add(key)
            unique.append((h, r, t))
    return unique


class Neo4jConnection:
    def __init__(self, uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def clear_graph(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def store_triples(self, triples):
        with self.driver.session() as session:
            for head, rel, tail in triples:
                session.run(
                    """
                    MERGE (h:Entity {name: $head})
                    MERGE (t:Entity {name: $tail})
                    MERGE (h)-[r:RELATION {type: $rel}]->(t)
                    """,
                    head=head, rel=rel, tail=tail,
                )

    def get_all_nodes(self):
        with self.driver.session() as session:
            result = session.run("MATCH (n:Entity) RETURN n.name AS name")
            return [record["name"] for record in result]

    def get_one_hop_triples(self, node_names, max_results=50):
        with self.driver.session() as session:
            results = []
            for name in node_names:
                result = session.run(
                    """
                    MATCH (h:Entity {name: $name})-[r]->(t:Entity)
                    RETURN h.name AS head, r.type AS relation, t.name AS tail
                    UNION
                    MATCH (h:Entity)-[r]->(t:Entity {name: $name})
                    RETURN h.name AS head, r.type AS relation, t.name AS tail
                    """,
                    name=name,
                )
                for record in result:
                    results.append((record["head"], record["relation"], record["tail"]))
            seen = set()
            unique_results = []
            for h, r, t in results:
                key = (h.lower(), r.lower(), t.lower())
                if key not in seen:
                    seen.add(key)
                    unique_results.append((h, r, t))
            return unique_results[:max_results]

    def get_shortest_paths(self, node_names, max_hops=4):
        if len(node_names) < 2:
            return []
        with self.driver.session() as session:
            all_paths = []
            pairs = [(node_names[i], node_names[j])
                     for i in range(len(node_names))
                     for j in range(i + 1, len(node_names))]
            for src, tgt in pairs:
                query = f"""
                    MATCH path = shortestPath(
                        (h:Entity {{name: $src}})-[*..{max_hops}]-(t:Entity {{name: $tgt}})
                    )
                    RETURN path
                    """
                result = session.run(query, src=src, tgt=tgt)
                for record in result:
                    path = record["path"]
                    nodes = [n["name"] for n in path.nodes]
                    rels = [r["type"] for r in path.relationships]
                    parts = []
                    for i, node in enumerate(nodes):
                        parts.append(f"({node})")
                        if i < len(rels):
                            parts.append(f"[{rels[i]}]")
                    all_paths.append(" -> ".join(parts))
            seen = set()
            unique_paths = []
            for p in all_paths:
                if p not in seen:
                    seen.add(p)
                    unique_paths.append(p)
            return unique_paths


class VectorStore:
    def __init__(self, path=CHROMA_DB_PATH):
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(
            name="kg_nodes",
            metadata={"hnsw:space": "cosine"},
        )
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

    def clear(self):
        existing = self.client.get_collection("kg_nodes")
        if existing.count() > 0:
            self.client.delete_collection("kg_nodes")
        self.collection = self.client.get_or_create_collection(
            name="kg_nodes",
            metadata={"hnsw:space": "cosine"},
        )

    def add_nodes(self, node_names):
        existing_ids = set(self.collection.get()["ids"]) if self.collection.count() > 0 else set()
        new_nodes = [n for n in node_names if n not in existing_ids]
        if not new_nodes:
            return
        embeddings = self.embedder.encode(new_nodes).tolist()
        self.collection.add(
            ids=new_nodes,
            embeddings=embeddings,
            metadatas=[{"name": n} for n in new_nodes],
        )

    def search(self, query, top_k=5):
        query_embedding = self.embedder.encode([query]).tolist()[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
        )
        matches = []
        if results["ids"]:
            for i, node_id in enumerate(results["ids"][0]):
                score = results["distances"][0][i]
                matches.append((node_id, 1 - score))
        return matches

    def count(self):
        return self.collection.count()


def build_knowledge_graph(client, neo4j_conn, vector_store, chunks):
    all_triples = []
    for i, chunk in enumerate(chunks):
        triples = extract_triples(chunk, client)
        all_triples.extend(triples)
    all_triples = deduplicate_triples(all_triples)
    neo4j_conn.clear_graph()
    neo4j_conn.store_triples(all_triples)
    nodes = neo4j_conn.get_all_nodes()
    vector_store.clear()
    vector_store.add_nodes(nodes)
    return all_triples, nodes
