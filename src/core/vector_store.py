import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever
from src.core.llm_provider import EmbeddingModelProvider


class VectorStoreManager:
    """
    A class that uses EmbeddingModelProvider to get an embedding model and
    then converts a PDF document into a searchable vector store.
    """

    def __init__(self, embedding_model_name: str = "text-embedding-ada-002"):
        """
        Initializes the VectorStoreManager.

        Args:
            embedding_model_name: The name of the OpenAI embedding model to use.
        """
        self.embedding_provider = EmbeddingModelProvider(embedding_model=embedding_model_name)
        self._embeddings: Embeddings | None = None

    async def _ensure_embeddings_loaded(self) -> Embeddings:
        """
        Ensures the embedding model is loaded asynchronously.
        """
        if self._embeddings is None:
            self._embeddings = await self.embedding_provider.async_get_embedding_model()
        return self._embeddings

    async def create_from_pdf(self, pdf_path: str) -> VectorStoreRetriever:
        """
        Asynchronously creates a temporary in-memory vector store from a given PDF file.

        Args:
            pdf_path: The file path to the PDF document.

        Returns:
            A LangChain VectorStoreRetriever instance ready for querying.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at {pdf_path}")

        # 1. Ensure embeddings are ready
        embeddings = await self._ensure_embeddings_loaded()

        # 2. Load the PDF document
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        # 3. Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        documents = text_splitter.split_documents(pages)

        # 4. Create an in-memory FAISS vector store and return a retriever
        vector_store = FAISS.from_documents(documents, embeddings)
        return vector_store.as_retriever()
