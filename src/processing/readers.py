from PIL import Image
from abc import abstractmethod

import fitz  # PyMuPDF

from surya.ocr import run_ocr as surya_ocr
from surya.model.detection import segformer as surya_ssegformer
from surya.model.recognition.model import load_model as surya_recognition_load_model
from surya.model.recognition.processor import (
    load_processor as surya_recognition_load_processor,
)


# Define a base class for extractions
class Extraction:
    def __init__(self, text: str, location: str) -> None:
        self.text: str = text  # The extracted text
        self.location: str = location  # The source of extraction (e.g., page)


class Reader:
    """
    Abstract class for reading data.
    """

    def __init__(self) -> None:
        """Initialize the Reader class."""

    @abstractmethod
    def get_text(self) -> list[Extraction]:
        """
        Abstract method to get the text from the file.

        Returns:
            list[Extraction]: List of extractions per page.
        """


###################################################################################
##################################### PDFs ########################################
###################################################################################
class PDFReader(Reader):
    """Class for reading PDFs."""

    def __init__(
        self, enable_ocr: bool = False, min_text_condifence: float = 0.35
    ) -> None:
        """Initialize the PDFReader class.

        Args:
            enable_ocr (bool, optional): Whether use OCR or not. Defaults to False.
            min_text_condifence (float, optional): Minimum confidence for the text. Defaults to 0.35.
        """
        super().__init__()
        if enable_ocr:  # Configure and load OCR models
            self.langs = ["en"]  # Replace with your languages
            self.det_processor = surya_ssegformer.load_processor()
            self.det_model = surya_ssegformer.load_model()
            self.rec_model = surya_recognition_load_model()
            self.rec_processor = surya_recognition_load_processor()

        self.enable_ocr = enable_ocr
        self.min_text_condifence = min_text_condifence

    def is_digital(self, pdf_path: str) -> bool:
        """
        Determine if a PDF is digital or scanned.

        Args:
            pdf_path (str): Path to the PDF file.

        Returns:
            bool: True if the PDF is digital, False if it's scanned.
        """
        # Check if the PDF contains selectable text
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                text = page.get_text()
                if text.strip():
                    return True
            return False
        except Exception as e:
            print("Error:", e)
            return False

    def get_text(self, pdf_path: str) -> list[Extraction]:
        """
        Implementation of get_text method for PDFReader.

        Args:
            pdf_path (str): Path to the PDF file.

        Returns:
            list[Extraction]: List of extractions.
        """
        if self.enable_ocr and not self.is_digital(pdf_path):
            return self.get_scanned_text(pdf_path)
        else:
            return self.get_digitized_text(pdf_path)

    def get_scanned_text(self, pdf_path: str) -> list[Extraction]:
        """
        Apply OCR to the PDF file.

        Args:
            pdf_path (str): Path to the PDF file.

        Returns:
            list[Extraction]: List of extractions.
        """
        doc = fitz.open(pdf_path)

        texts = []
        for page_index, page in enumerate(doc):
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            predictions = surya_ocr(
                [img],
                [self.langs],
                self.det_model,
                self.det_processor,
                self.rec_model,
                self.rec_processor,
            )

            """
            The predictions is a list of OCRResult(s), an object
            that contains TextLine(s) with the following attributes:
            - polygon: list of coordinates of the bounding box
            - confidence: confidence score of the prediction
            - text: the predicted text
            - bbox: bounding box of the text line
            """
            image_predicition = predictions[0]  # We only used one image
            text = "\n".join(
                [
                    line.text
                    for line in image_predicition.text_lines
                    if line.confidence > self.min_text_condifence
                ]
            )

            texts.append(
                Extraction(
                    text=text,
                    location=f"Page {page_index + 1}",
                )
            )

        return texts

    def get_digitized_text(self, pdf_path: str) -> list[Extraction]:
        """
        Get digitized text from the PDF file.

        Args:
            pdf_path (str): Path to the PDF file.

        Returns:
            list[Extraction]: List of extractions.
        """
        # Extract text from digital PDF
        texts = []
        doc = fitz.open(pdf_path)
        for page_index, page in enumerate(doc):
            texts.append(
                Extraction(
                    text=page.get_text(),
                    location=f"Page {page_index + 1}",
                )
            )

        return texts
