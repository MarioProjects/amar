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
    def ask(
        self, query: str | list[dict], query_context: list[dict] = None
    ) -> LLMResponse:
        """
        Abstract method to ask the model a question.

        Args:
            query (str | list[dict]): Simple text or list of dictionaries with role and content.
            query_context (list[dict], optional): Context of the query. Defaults to None.

        Returns:
            LLMResponse: Response from the model.
        """
        pass
