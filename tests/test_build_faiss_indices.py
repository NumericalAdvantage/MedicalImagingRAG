import os
import json
import tempfile
import numpy as np
import faiss
import pytest

from app.build_faiss_indices import build_index

def test_build_index_text(tmp_path):
    # Create a fake JSONL file with text embeddings
    jsonl_path = tmp_path / "embeddings.jsonl"
    data = [
        {"text_embedding": [0.1, 0.2, 0.3]},
        {"text_embedding": [0.4, 0.5, 0.6]},
    ]
    with open(jsonl_path, "w") as f:
        for obj in data:
            f.write(json.dumps(obj) + "\n")
    index_path = tmp_path / "faiss_text.index"
    build_index(str(jsonl_path), "text_embedding", str(index_path))
    index = faiss.read_index(str(index_path))
    assert index.d == 3
    assert index.ntotal == 2

def test_build_index_image_flatten(tmp_path):
    # Create a fake JSONL file with image embeddings (2D arrays)
    jsonl_path = tmp_path / "embeddings.jsonl"
    data = [
        {"image_embedding": [[0.1, 0.2], [0.3, 0.4]]},
        {"image_embedding": [[0.5, 0.6], [0.7, 0.8]]},
    ]
    with open(jsonl_path, "w") as f:
        for obj in data:
            f.write(json.dumps(obj) + "\n")
    index_path = tmp_path / "faiss_image.index"
    build_index(str(jsonl_path), "image_embedding", str(index_path), flatten=True)
    index = faiss.read_index(str(index_path))
    assert index.d == 4
    assert index.ntotal == 2 