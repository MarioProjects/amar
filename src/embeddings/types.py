from abc import abstractmethod


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
