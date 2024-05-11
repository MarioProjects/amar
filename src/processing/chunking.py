from abc import abstractmethod

from src.processing.readers import Extraction


class Chunk:
    def __init__(self, text: str, location: str) -> None:
        """
        Represents a chunk of text extracted from the original text.

        Args:
            text (str): The text chunk.
            location (str): The source of extraction (e.g., page).
        """
        self.text: str = text  # The text chunk
        self.location: str = location  # The source of extraction (e.g., page)


class Chunker:
    """
    Abstract class for chunking text.
    """

    def __init__(self) -> None:
        """Initialize the Chunker class."""

    @abstractmethod
    def get_chunks(self) -> list[Chunk]:
        """
        Abstract method to get the text chunks.

        Returns:
            List[Chunk]: Text chunks.
        """
        pass


###################################################################################
##################################### PDFs ########################################
###################################################################################
class SymbolChunker(Chunker):
    """Class for chunking text based on breaks."""

    def __init__(
        self, chars_limit: int = 1024, overlap: int = 256, symbol: str = "\n"
    ) -> None:
        """Initialize the BreaksChunker class.

        Args:
            chars_limit (int, optional): The maximum number of characters in a chunk. Defaults to 256.
            overlap (int, optional): The number of characters to overlap between chunks. Defaults to 50.
            symbol (str, optional): The symbol to use as a break. Defaults to "\n".
        """
        super().__init__()
        self.chars_limit: int = chars_limit
        self.overlap: int = overlap
        self.symbol: str = symbol

    def get_chunks(
        self,
        extractions: Extraction,
    ) -> list[Chunk]:
        """
        Get the text chunks based on breaks.

        Args:
            extractions (Extraction): The text to chunk.


        Returns:
            List[Chunk]: Text chunks.
        """
        chunks = []  # List to store the chunks of text

        # Iterate through each Extraction object
        for extraction in extractions:
            lines = extraction.text.split(self.symbol)  # Split the text into lines

            current_chunk = (
                ""  # Initialize an empty string for the current chunk of text
            )
            current_chunk_length = 0  # Initialize the length of the current chunk

            # Iterate through each line of the text
            for line in lines:
                # If adding the current line to the current chunk doesn't exceed the text limit
                if len(current_chunk) + len(line) + self.overlap <= self.chars_limit:
                    current_chunk += line + "\n"  # Add the line to the current chunk
                    current_chunk_length += (
                        len(line) + 1
                    )  # Update the length of the current chunk
                else:  # If adding the current line to the current chunk exceeds the text limit
                    chunks.append(
                        Chunk(current_chunk.strip(), extraction.location)
                    )  # Add the current chunk to the list
                    current_chunk = (
                        current_chunk[len(line) - self.overlap :] + line + "\n"
                    )  # Start a new chunk with overlap
                    current_chunk_length = (
                        len(line) + 1
                    )  # Update the length of the new chunk

            # Add the remaining part as a chunk if it exceeds the text limit
            if current_chunk:
                chunks.append(Chunk(current_chunk.strip(), extraction.location))

        return chunks
