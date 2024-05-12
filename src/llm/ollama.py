import requests
import ollama

from src.helpers.ollama import OllamaHelper
from src.llm.types import LLMModel, LLMResponse


class OllamaLLM(LLMModel):
    """Class for Ollama LLMs."""

    def __init__(self, model: str, base_url: str = "http://localhost:11434") -> None:
        super().__init__(model=model, base_url=base_url)
        self.helper = OllamaHelper(base_url=base_url, verbose=True)
        self.helper.pull_model(model)

    def ask(self, query: str) -> LLMResponse:
        """
        Implementation of get_embedding method for OllamaEmbedding.

        Args:
            query (str):

        Returns:
            Embedding: Embedding of the input text.
        """
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": query,
                    },
                ],
            )
            return response["message"]["content"]
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error connecting to Ollama model: {e}") from e
        except Exception as e:
            raise Exception(f"Error getting response from Ollama model: {e}") from e
