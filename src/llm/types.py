from abc import abstractmethod


LLMResponse = str


class LLMModel:
    """
    Abstract class for LLM models.
    """

    def __init__(self, model: str, base_url: str) -> None:
        """Initialize the LLMModel class.

        Args:
            model (str): Name of the embedding model.
            base_url (str): Base url the model is hosted by Ollama.
        """
        self.model: str = model
        self.base_url: str = base_url

    @abstractmethod
    def get_embedding(self, query: str) -> LLMResponse:
        """
        Abstract method to ask the model a question.

        Args:
            query (str): Input text to generate the response from.

        Returns:
            LLMResponse: Response of the input text.
        """
