"""PDF Q&A agent: ingest PDFs, break them into sections, answer questions with citations."""

from __future__ import annotations

from .config import Settings, get_api_key, get_settings

__all__ = ["build_services", "Services", "Settings", "get_settings", "get_api_key"]


class MissingAPIKeyError(RuntimeError):
    pass


class Services:
    """Store, ingestor and agent built from one Settings object."""

    def __init__(self, settings: Settings, api_key: str):
        from .agent import PDFAgent
        from .ingestion import Ingestor
        from .llm import OpenAIChatModel, OpenAIEmbedder
        from .store import DocumentStore

        self.settings = settings
        embedder = OpenAIEmbedder(api_key, settings.embedding_model, settings.request_timeout_s)
        self.store = DocumentStore(settings.data_dir, embedder, settings.collection_name)
        self.ingestor = Ingestor(self.store, settings)
        llm = OpenAIChatModel(
            api_key, settings.chat_model, settings.temperature, settings.max_answer_tokens, settings.request_timeout_s
        )
        self.agent = PDFAgent(self.store, llm, settings)


def build_services(settings: Settings | None = None) -> Services:
    settings = settings or get_settings()
    key = get_api_key()
    if key is None:
        raise MissingAPIKeyError(
            "No OpenAI API key found. Copy .env.example to .env and set OPENAI_API_KEY, "
            "or set the OPENAI_API_KEY environment variable."
        )
    return Services(settings, key)
