from langchain.schema import Document
from langchain.text_splitter import TextSplitter
import re

class CustomTextSplitter(TextSplitter):
    def __init__(self):
        super().__init__()

    def split_text_with_metadata(self, text: str) -> list[Document]:
        """
        Splits text into chunks starting with 'XXX. člen' at the beginning of a line.
        Returns a list of Documents with lowercase content, original text, and člen number.
        """
        # Match lines that start with a number followed by '. člen'
        pattern = re.compile(r'^\s*(?P<clen>\d+)\. člen', re.MULTILINE)
        matches = list(pattern.finditer(text))

        documents = []
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i+1].start() if i + 1 < len(matches) else len(text)
            original_chunk = text[start:end].strip()
            clen_number = match.group("clen")

            doc = Document(
                page_content=original_chunk.lower(),
                metadata={
                    "original_text": original_chunk,
                    "clen": clen_number
                }
            )
            documents.append(doc)

        return documents
