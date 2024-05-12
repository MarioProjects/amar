import requests

from src.helpers.ollama import OllamaHelper
from src.embeddings.types import EmbeddingModel, Embedding


class OllamaEmbedding(EmbeddingModel):
    """Class for Ollama embeddings."""

    def __init__(self, model: str, base_url: str = "http://localhost:11434") -> None:
        super().__init__(model=model, base_url=base_url)
        self.helper = OllamaHelper(base_url=base_url, verbose=True)
        self.helper.pull_model(model)

    def get_embedding(self, text: str) -> Embedding:
        """
        Implementation of get_embedding method for OllamaEmbedding.

        Args:
            text (str): Input text to generate the embedding from.

        Returns:
            Embedding: Embedding of the input text.
        """
        ollama_request_body = {"prompt": text, "model": self.model}

        response = requests.post(
            url=f"{self.base_url}/api/embeddings",
            headers={"Content-Type": "application/json"},
            json=ollama_request_body,
        )
        response.encoding = "utf-8"

        if response.status_code != 200:
            optional_detail = response.json().get("error")
            raise ValueError(
                f"Ollama embedding call failed. Status code: {response.status_code}."
                f" Details: {optional_detail}"
            )

        try:
            return response.json()["embedding"]

        except requests.exceptions.JSONDecodeError as e:
            raise ValueError(
                f"Error raised for Ollama Call: {e}.\nResponse: {response.text}"
            )
