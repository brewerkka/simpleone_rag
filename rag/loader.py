from pathlib import Path
from typing import List
import json
from langchain.schema import Document


def load_chunks(*paths: str) -> List[Document]:

    documents: List[Document] = []
    for file_path in paths:
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Файл не найден: {path}")

        raw = json.loads(path.read_text(encoding="utf-8"))
        for chunk in raw.get("chunks", []):
            documents.append(
                Document(page_content=chunk["text"], metadata=chunk.get("metadata", {}))
            )
    return documents
