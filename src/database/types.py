import uuid
from abc import abstractmethod
from src.embeddings.types import Embedding


class CollectionItem:
    _used_ids = set()  # Set to store used IDs

    def __init__(
        self,
        text: str,
        embedding: Embedding,
        document_path: str,
        location: str,
        id: str = None,
    ) -> None:
        """Initialize the CollectionItem class.

        Args:
            text (str): Text of the item.
            embedding (Embedding): Embedding of the item.
            document_path (str): Path to the document.
            location (str): Location of the item.
            id (str, optional): ID of the item. Defaults to None.
        """
        if id:
            self.id: str = id
        else:
            self.id: str = self._generate_unique_id()  # Generate a unique ID
        self.embedding: Embedding = embedding
        self.document_path: str = document_path
        self.location: str = location
        self.text: str = text

    def _generate_unique_id(self) -> str:
        """Generate a unique ID for the CollectionItem."""
        while True:
            new_id = str(uuid.uuid4())
            if new_id not in self._used_ids:
                self._used_ids.add(new_id)
                return new_id


class VectorDatabase:
    """
    Abstract class for vector databases.
    """

    def __init__(self, path: str, name: str) -> None:
        """Initialize the Database class.

        Args:
            path (str): Path to the database.
            name (str): Name of the database.
        """
        self.path: str = path
        self.name: str = name
        self.db: any = None

    @abstractmethod
    def create(self) -> None:
        """
        Abstract method to create the database.
        """

    @abstractmethod
    def load(self) -> None:
        """
        Abstract method to load the database.
        """

    @abstractmethod
    def remove(self) -> None:
        """
        Abstract method to remove the database.
        """

    @abstractmethod
    def insert(self, document: CollectionItem) -> None:
        """
        Abstract method to insert a document into the database.

        Args:
            document (CollectionItem): Document to insert.
        """

    @abstractmethod
    def search(self, query: dict) -> list[CollectionItem]:
        """
        Abstract method to search for documents in the database.

        Args:
            query (dict): Query to search for.

        Returns:
            List[CollectionItem]: List of documents that match the query.
        """
