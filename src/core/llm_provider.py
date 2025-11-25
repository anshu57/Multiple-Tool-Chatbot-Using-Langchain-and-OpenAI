import os
import asyncio
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings


class ChatModelProvider:
    """A class to initialize and provide an OpenAI Chat Model instance."""

    def __init__(
        self, model: str = "gpt-4-turbo-preview", temperature: float = 0.0
    ):
        """
        Initializes the Chat Model provider.

        Args:
            model: The name of the OpenAI model to use.
            temperature: The temperature for the model's responses.
        """
        self.model = model
        self.temperature = temperature
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")

        self._llm: BaseChatModel | None = None

    @classmethod
    async def async_init(
        cls, model: str = "gpt-4-turbo-preview", temperature: float = 0.0
    ) -> "ChatModelProvider":
        """
        Asynchronous factory for creating a ChatModelProvider instance.
        This allows for a consistent async initialization pattern.
        """
        return cls(model=model, temperature=temperature)

    def get_llm(self) -> BaseChatModel:
        """
        Initializes and returns an instance of the ChatOpenAI model.
        The instance is cached for subsequent calls.
        """
        if self._llm is None:
            self._llm = ChatOpenAI(model=self.model, temperature=self.temperature, api_key=self.api_key)
        return self._llm

    async def async_get_llm(self) -> BaseChatModel:
        """
        Asynchronously initializes and returns an instance of the ChatOpenAI model.
        The instance is cached for subsequent calls.
        """
        if self._llm is None:
            self._llm = ChatOpenAI(model=self.model, temperature=self.temperature, api_key=self.api_key)
        return self._llm


class EmbeddingModelProvider:
    """A class to initialize and provide an OpenAI Embedding Model instance."""

    def __init__(self, embedding_model: str = "text-embedding-ada-002"):
        """
        Initializes the Embedding Model provider.

        Args:
            embedding_model: The name of the OpenAI embedding model to use.
        """
        self.embedding_model_name = embedding_model
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")

        self._embedding_model: Embeddings | None = None

    @classmethod
    async def async_init(cls, embedding_model: str = "text-embedding-ada-002") -> "EmbeddingModelProvider":
        """Asynchronous factory for creating an EmbeddingModelProvider instance."""
        return cls(embedding_model=embedding_model)

    def get_embedding_model(self) -> Embeddings:
        """
        Initializes and returns an instance of the OpenAIEmbeddings model.
        The instance is cached for subsequent calls.
        """
        if self._embedding_model is None:
            self._embedding_model = OpenAIEmbeddings(model=self.embedding_model_name, openai_api_key=self.api_key)
        return self._embedding_model

    async def async_get_embedding_model(self) -> Embeddings:
        """
        Asynchronously initializes and returns an instance of the OpenAIEmbeddings model.
        The instance is cached for subsequent calls.
        """
        if self._embedding_model is None:
            self._embedding_model = OpenAIEmbeddings(model=self.embedding_model_name, openai_api_key=self.api_key)
        return self._embedding_model

 