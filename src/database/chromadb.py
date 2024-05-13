import chromadb

from src.embeddings.types import Embedding
from src.database.types import CollectionItem, VectorDatabase


class ChromaDB(VectorDatabase):
    def __init__(self, path: str = "chromadb", name: str = "documents") -> None:
        """Initialize the ChromaDB class.

        Args:
            path (str): Path to the database.
            name (str): Name of the database.
        """
        super().__init__(path=path, name=name)
        self.load_or_create()

    @property
    def num_documents(self) -> int:
        """Get the number of documents in the database."""
        return self.collection.count()

    def load_or_create(self) -> None:
        """Load the database."""
        self.db_client = chromadb.PersistentClient(path=self.path)
        # Get a collection object from an existing collection, by name.
        # If it doesn't exist, create it.
        self.collection = self.db_client.get_or_create_collection(name=self.name)

    def remove(self) -> None:
        """Remove the database collection."""
        self.db_client.delete_collection(name=self.name)
        self.load_or_create()  # Recreate empty the collection

    def insert(self, document: CollectionItem) -> None:
        """
        Insert a document into the database.

        Args:
            document (CollectionItem): Document to insert.
        """
        self.collection.add(
            embeddings=document.embedding,
            metadatas={
                "document_path": document.document_path,
                "location": document.location,
                "text": document.text,
            },
            ids=[document.id],
        )

    def search(self, query_embedding: Embedding, top_k: int) -> list[CollectionItem]:
        """
        Search for the top k similar documents.

        Args:
            query (Embedding): Query embedding to search for.
            top_k (int): Number of similar documents to return.

        Returns:
            List[CollectionItem]: List of similar documents.
        """
        results = self.collection.query(
            query_embedding, n_results=top_k, include=["metadatas", "embeddings"]
        )

        items = []
        for id, metadata, embedding in zip(
            results["ids"][0], results["metadatas"][0], results["embeddings"][0]
        ):
            items.append(
                CollectionItem(
                    id=id,
                    text=metadata["text"],
                    embedding=Embedding(embedding),
                    document_path=metadata["document_path"],
                    location=metadata["location"],
                )
            )

        return items
