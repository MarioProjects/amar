import requests
import ollama

from src.database.types import CollectionItem
from src.helpers.ollama import OllamaHelper
from src.llm.types import LLMModel, LLMResponse


class OllamaLLM(LLMModel):
    """Class for Ollama LLMs."""

    def __init__(self, model: str, base_url: str = "http://localhost:11434") -> None:
        super().__init__(model=model, base_url=base_url)
        self.helper = OllamaHelper(base_url=base_url, verbose=True)
        self.helper.pull_model(model)

    def ask(
        self, query: str | list[dict], query_context: list[CollectionItem] = None
    ) -> LLMResponse:
        """
        Implementation of get_embedding method for OllamaEmbedding.

        Args:
            query (str | list[dict]): Simple text or list of dictionaries with role and content.
            query_context (list[CollectionItem], optional): Context of the query. Defaults to None.

        Returns:
            LLMResponse: Response from the model.
        """
        try:
            if isinstance(query, str):
                query = [{"role": "user", "content": query}]

            if query_context is not None:
                # Add the information to the last user query
                context_info = "Next is the context information:\n\n"
                context_info += "\n\n".join([item.text for item in query_context])
                query[-1]["content"] += context_info

            response = ollama.chat(model=self.model, messages=query)
            return response["message"]["content"]
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error connecting to Ollama model: {e}") from e
        except Exception as e:
            raise Exception(f"Error getting response from Ollama model: {e}") from e
