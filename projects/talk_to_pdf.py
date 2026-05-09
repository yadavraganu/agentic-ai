from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
import os


def load_pdf_as_messages(
    pdf_path: str, page_range: tuple[int, int]
) -> list[BaseMessage]:
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()[page_range[0] : page_range[1] + 1]
    messages = []
    for doc in documents:
        content = doc.page_content
        messages.append(content)
    return messages


if __name__ == "__main__":
    file_path = os.path.abspath("DW_Book.pdf")
    print(f"Loading PDF from {file_path}")
    messages = load_pdf_as_messages(file_path, (6, 6))  # Load first 3 pages
    print(f"Loaded {len(messages)} messages from PDF")
    print(messages)
