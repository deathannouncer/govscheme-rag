import os

from dotenv import load_dotenv

load_dotenv()

POSTGRES_DSN = os.environ["POSTGRES_DSN"]
MILVUS_HOST = os.environ["MILVUS_HOST"]
MILVUS_PORT = os.environ["MILVUS_PORT"]
REDIS_HOST = os.environ["REDIS_HOST"]
REDIS_PORT = int(os.environ["REDIS_PORT"])

EMBEDDING_MODEL = os.environ["EMBEDDING_MODEL"]
RERANKER_MODEL = os.environ["RERANKER_MODEL"]
OLLAMA_MODEL = os.environ["OLLAMA_MODEL"]

TOP_K_VECTOR = int(os.environ["TOP_K_VECTOR"])
TOP_K_FULLTEXT = int(os.environ["TOP_K_FULLTEXT"])
TOP_K_FUSED = int(os.environ["TOP_K_FUSED"])
TOP_K_FINAL = int(os.environ["TOP_K_FINAL"])
MAX_AGENT_LOOPS = int(os.environ["MAX_AGENT_LOOPS"])
CACHE_SIMILARITY_THRESHOLD = float(os.environ["CACHE_SIMILARITY_THRESHOLD"])

COLLECTION_NAME = "scheme_chunks"
