"""Builds FAISS indices for text and image embeddings from a JSONL file.

This script reads embeddings from a JSONL file and creates FAISS indices for
fast similarity search, saving them to disk.
"""

import os
import json
from typing import Any, List

import numpy as np
import faiss

os.makedirs("embeddings", exist_ok=True)

def build_index(jsonl_path: str, field: str, output_path: str, flatten: bool = False) -> None:
    """Build a FAISS index from embeddings in a JSONL file.

    Args:
        jsonl_path: Path to the JSONL file containing embeddings.
        field: The field in each JSON object to use as the embedding.
        output_path: Path to save the FAISS index.
        flatten: Whether to flatten the embedding vectors (for images).
    """
    vectors: List[Any] = []
    with open(jsonl_path) as f:
        for line in f:
            obj = json.loads(line)
            vec = obj[field]
            if flatten:
                vec = np.array(vec).flatten().tolist()
            vectors.append(vec)
    if not vectors:
        raise ValueError("No vectors found in the JSONL file.")
    dim = len(vectors[0])
    index = faiss.IndexFlatL2(dim)
    np_vectors = np.array(vectors, dtype=np.float32)
    index.add(np_vectors)
    faiss.write_index(index, output_path)
    print(f"Index saved to {output_path}")


def main() -> None:
    """Build both text and image FAISS indices from embeddings JSONL."""
    build_index("embeddings/embeddings.jsonl", "text_embedding", "embeddings/faiss_text.index")
    build_index("embeddings/embeddings.jsonl", "image_embedding", "embeddings/faiss_image.index", flatten=True)


if __name__ == "__main__":
    main()
