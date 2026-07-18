import os

from openai import OpenAI
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("HYDRA_API_KEY")
OPENAI_BASE_URL = os.getenv(
    "OPENAI_BASE_URL",
    os.getenv("HYDRA_BASE_URL", "https://api.hydraai.ru/v1"),
).rstrip("/")
OPENAI_MODEL = os.getenv(
    "OPENAI_MODEL",
    "gpt-5-mini",
)
OPENAI_PREPROCESSOR_MODEL = os.getenv("OPENAI_PREPROCESSOR_MODEL")
OPENAI_EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL",
    "text-embedding-3-small",
)

def get_chatGPT_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )
    return client


def get_langchain_openai_chat_model(model: str | None = None) -> ChatOpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return ChatOpenAI(
        api_key=SecretStr(OPENAI_API_KEY),
        base_url=OPENAI_BASE_URL,
        model=model or OPENAI_MODEL,
    )


def get_preprocessor_chat_model() -> ChatOpenAI:
    return get_langchain_openai_chat_model(
        model=OPENAI_PREPROCESSOR_MODEL or OPENAI_MODEL
    )


def get_openai_embedding(text: str) -> list[float]:
    """Return an embedding vector using the configured OpenAI-compatible client."""

    client = get_chatGPT_client()
    response = client.embeddings.create(
        model=OPENAI_EMBEDDING_MODEL,
        input=text,
    )
    return list(response.data[0].embedding)


def get_openai_embeddings(texts: list[str]) -> list[list[float]]:
    """Return embedding vectors for a batch of texts."""

    if not texts:
        return []
    client = get_chatGPT_client()
    response = client.embeddings.create(
        model=OPENAI_EMBEDDING_MODEL,
        input=texts,
    )
    return [list(item.embedding) for item in response.data]
