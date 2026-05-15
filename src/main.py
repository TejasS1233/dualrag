import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from src.config import (
    GROQ_API_KEY, GROQ_BASE_URL, LLM_MODEL,
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
    CHROMA_DB_PATH,
)
from src.kg_construction import (
    chunk_corpus, build_knowledge_graph,
    Neo4jConnection, VectorStore,
)
from src.query_enhancer import enhance_query
from src.retriever import retrieve
from src.qa_pipeline import generate_answer


def build(client):
    corpus_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "corpus.txt")
    if not os.path.exists(corpus_path):
        print(f"Corpus not found at {corpus_path}")
        return
    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read()
    chunks = chunk_corpus(text)
    neo4j_conn = Neo4jConnection()
    vector_store = VectorStore()
    triples, nodes = build_knowledge_graph(client, neo4j_conn, vector_store, chunks)
    print(f"Built KG: {len(nodes)} nodes, {len(triples)} triples")
    neo4j_conn.close()


def ask(client, query):
    neo4j_conn = Neo4jConnection()
    vector_store = VectorStore()
    print("Enhancing query...")
    query_nodes = enhance_query(query, vector_store, neo4j_conn, client)
    explicit_nodes, enhanced_nodes, implicit_nodes = query_nodes
    print(f"  Explicit nodes: {explicit_nodes}")
    print(f"  Enhanced nodes: {enhanced_nodes}")
    print(f"  Implicit nodes: {implicit_nodes}")
    print("Retrieving from KG...")
    one_hop_triples, shortest_paths = retrieve(query_nodes, neo4j_conn)
    print(f"  Triples: {len(one_hop_triples)}")
    print(f"  Shortest paths: {len(shortest_paths)}")
    print("Generating answer...")
    answer = generate_answer(query, one_hop_triples, shortest_paths, client)
    print(f"\nQuery: {query}")
    print(f"Answer: {answer}")
    neo4j_conn.close()
    return answer


def main():
    if not GROQ_API_KEY:
        print("Error: GROQ_API_KEY not set. Add it to .env file.")
        sys.exit(1)
    client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)
    if len(sys.argv) < 2:
        print("Usage: uv run python src/main.py <command> [args]")
        print("Commands:")
        print("  build              Build KG from corpus")
        print("  ask <question>     Ask a question")
        sys.exit(1)
    command = sys.argv[1]
    if command == "build":
        build(client)
    elif command == "ask":
        if len(sys.argv) < 3:
            print("Usage: uv run python src/main.py ask <question>")
            sys.exit(1)
        query = " ".join(sys.argv[2:])
        ask(client, query)
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
