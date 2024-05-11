import requests
from abc import abstractmethod

from src.helpers.ollama import OllamaHelper


Embedding = list[float]


class EmbeddingModel:
    """
    Abstract class for embedding models.
    """

    def __init__(self, model: str, base_url: str) -> None:
        """Initialize the EmbeddingModel class.

        Args:
            model (str): Name of the embedding model.
            base_url (str): Base url the model is hosted by Ollama.
        """
        self.model: str = model
        self.base_url: str = base_url

    @abstractmethod
    def get_embedding(self, text: str) -> Embedding:
        """
        Abstract method to get the embedding of the model.

        Args:
            text (str): Input text to generate the embedding from.

        Returns:
            Embedding: Embedding of the input text.
        """


###################################################################################
################################# OllamaEmbedding #################################
###################################################################################
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
