import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
LLM_MODEL = "llama-3.3-70b-versatile"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "dualrag_password")
CHROMA_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "chroma_db")
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 20
MAX_IMPLICIT_NODES = 3
MAX_SHORTEST_PATH_HOPS = 4
TOP_K_EMBEDDING_MATCH = 5
EMBEDDING_SIMILARITY_THRESHOLD = 0.5
