from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
import requests, os
import os
import asyncio
from typing import List, Dict, Any, Type
from langchain_core.vectorstores import VectorStoreRetriever
from pydantic import PrivateAttr
from langchain_core.documents import Document
from src.core.logger import get_logger

logger = get_logger(__name__)
class SearchTool:
    def __init__(self):
        # keep the duckduckgo tool instance on the object
        self.tool = DuckDuckGoSearchRun(region="us-en")


class StockPriceTool(BaseTool):
    """LangChain-compatible tool to fetch latest stock price from Alpha Vantage.

    Call .run(symbol) to use synchronously, or the tool can be passed into an agent.
    """
    name: str = "stock_price"
    description: str = (
        "Fetch latest stock price for a given ticker symbol using Alpha Vantage. "
        "Input: a ticker symbol string (e.g. 'AAPL'). Output: JSON response from Alpha Vantage."
    )

    api_key: str | None = None

    def model_post_init(self, __context: Any) -> None:
        """Get the API key from the environment if it's not provided."""
        self.api_key = self.api_key or os.environ.get("ALPHAVANTAGE_API_KEY")

    def _run(self, symbol: str) -> dict: 
        if not symbol or not isinstance(symbol, str):
            raise ValueError("symbol must be a non-empty string")

        url = (
            f"https://www.alphavantage.co/query"
            f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey={self.api_key}"
        )
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json()

    async def _arun(self, symbol: str) -> dict:
        # For simplicity, run sync implementation in async context
        return self._run(symbol)


class RAGInput(BaseModel):
    """Input schema for the RAGTool."""
    query: str = Field(description="The user's query for searching the document.")
    # The retriever is passed in by the manager, not the LLM, so it's a private attribute.
    _retriever: VectorStoreRetriever = PrivateAttr()

class RAGTool(BaseTool):
    """Tool for retrieving information from a document associated with a thread."""
    name: str = "rag_tool"
    description: str = (
        "Retrieve relevant information from the pdf document. "
        "Use this tool when the user asks factual/conceptual questions "
        "that might be answered from the stored documents."
    )
    args_schema: Type[BaseModel] = RAGInput

    def _run(self, query: str, **kwargs: Any) -> dict:
        """Use the tool synchronously."""
        retriever = kwargs["retriever"]
        result = retriever.invoke(query)
        context = [doc.page_content for doc in result]
        metadata = [doc.metadata for doc in result]

        return {
            'query': query,
            'context': context,
            'metadata': metadata
        }

    async def _arun(self, query: str, **kwargs: Any) -> dict:
        """Use the tool asynchronously."""
        logger.info(f"RAGTool searching for: '{query}'")
        retriever = kwargs["retriever"]
        result: List[Document] = await retriever.ainvoke(query)
        context = [doc.page_content for doc in result]
        metadata = [doc.metadata for doc in result]

        return {
            'query': query,
            'context': context,
            'metadata': metadata
        }
