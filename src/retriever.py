from src.kg_construction import Neo4jConnection


def retrieve(query_nodes, neo4j_conn):
    explicit_nodes, enhanced_nodes, implicit_nodes = query_nodes
    all_nodes = list(set(explicit_nodes + enhanced_nodes + implicit_nodes))
    if not all_nodes:
        return [], []
    one_hop_triples = neo4j_conn.get_one_hop_triples(all_nodes)
    shortest_paths = neo4j_conn.get_shortest_paths(all_nodes)
    return one_hop_triples, shortest_paths
