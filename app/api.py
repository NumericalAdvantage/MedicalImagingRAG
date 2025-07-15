"""FastAPI backend for Medical Imaging RAG system.

Provides endpoints for text-based and image-based search using DICOM embeddings.
"""

import os
import json
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
import pydicom
import requests
import torch
from fastapi import FastAPI, UploadFile, File
from monai.transforms.compose import Compose
from monai.transforms.spatial.array import Resize
from monai.transforms.intensity.array import ScaleIntensity
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI()

TEXT_EMBED_MODEL_NAME = os.environ.get(
    'TEXT_EMBED_MODEL',
    'pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb')
text_model = SentenceTransformer(TEXT_EMBED_MODEL_NAME)


def load_medical_model() -> tuple:
    """Load BioMedCLIP vision and text encoders, fallback to DenseNet if needed.

    Returns:
        Tuple of (model, preprocess, tokenizer) or fallback values.
    """
    try:
        from open_clip import create_model_from_pretrained, get_tokenizer
        model, preprocess = create_model_from_pretrained(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        tokenizer = get_tokenizer(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        return model, preprocess, tokenizer
    except Exception as exc:
        print(f"Could not load BioMedCLIP: {exc}")
        print("Falling back to DenseNet121 (chest X-ray specific)")
        import torchvision.models as models
        vision_model = models.densenet121(pretrained=True)
        vision_model.features.conv0 = torch.nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        vision_model = torch.nn.Sequential(*list(vision_model.children())[:-1])
        return vision_model, None, None


try:
    BIOMEDCLIP_MODEL, BIOMEDCLIP_PREPROCESS, BIOMEDCLIP_TOKENIZER = load_medical_model()
    if BIOMEDCLIP_MODEL is not None:
        BIOMEDCLIP_MODEL.eval()
    print("Loaded BioMedCLIP model (OpenCLIP)")
except Exception as exc:
    print(f"Could not load BioMedCLIP model: {exc}")
    BIOMEDCLIP_MODEL = None
    BIOMEDCLIP_PREPROCESS = None
    BIOMEDCLIP_TOKENIZER = None

IMG_TRANSFORM = Compose([
    Resize((224, 224)),
    ScaleIntensity()
])

# Load embeddings and indices
try:
    TEXT_INDEX = faiss.read_index("embeddings/faiss_text.index")
    IMAGE_INDEX = faiss.read_index("embeddings/faiss_image.index")
    with open("embeddings/embeddings.jsonl") as f:
        EMBEDDINGS = [json.loads(line) for line in f]
    print("Loaded embeddings and indices")
except Exception as exc:
    print(f"Could not load embeddings: {exc}")
    TEXT_INDEX = None
    IMAGE_INDEX = None
    EMBEDDINGS = []


class Query(BaseModel):
    """Request model for text-based RAG query."""
    query: str


@app.post("/rag_query")
async def rag_query(q: Query) -> Dict[str, Any]:
    """Text-based RAG query endpoint.

    Args:
        q: Query object containing the user's question.

    Returns:
        Dict with context, context_items, and answer.
    """
    try:
        if TEXT_INDEX is None or not EMBEDDINGS:
            return {
                "error": "Embeddings not loaded",
                "context": "",
                "answer": "Please ensure embeddings are generated first."
            }
        if BIOMEDCLIP_MODEL is None or BIOMEDCLIP_TOKENIZER is None:
            return {
                "error": "Model or tokenizer not loaded",
                "context": "",
                "answer": "Please ensure the model and tokenizer are loaded."
            }
        tokens = BIOMEDCLIP_TOKENIZER([q.query], context_length=256)
        with torch.no_grad():
            q_embed = BIOMEDCLIP_MODEL.encode_text(tokens).squeeze().cpu().numpy().astype("float32")
        _, indices = TEXT_INDEX.search(np.array([q_embed]), k=10)
        context_items = [EMBEDDINGS[i] for i in indices[0]]
        context = "\n\n".join(item["metadata_text"] for item in context_items)

        prompt = f"""
        You are a radiologist AI assistant. Given the findings below, answer the question:

        Findings:
        {context}

        Question:
        {q.query}

        Answer:
        """
        try:
            res = requests.post(
                "http://ollama:11434/api/generate",
                json={
                    "model": "llama2:7b-chat",
                    "prompt": prompt,
                    "stream": False
                },
                timeout=300
            )
            response_data = res.json()
            if "response" in response_data:
                answer = response_data["response"]
            elif "content" in response_data:
                answer = response_data["content"]
            else:
                answer = str(response_data)
        except requests.exceptions.Timeout:
            answer = (
                "The language model is taking too long to respond. "
                "Please try a simpler question or try again later."
            )
        except Exception as exc:
            answer = f"Error calling language model: {str(exc)}"
        return {
            "context": context,
            "context_items": context_items,
            "answer": answer
        }
    except Exception as exc:
        return {
            "error": str(exc),
            "context": "",
            "answer": "An error occurred while processing your query."
        }


@app.post("/image_search")
async def image_search(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Image-based search endpoint for similar DICOM images.

    Args:
        file: Uploaded DICOM file.

    Returns:
        Dict with search results or error message.
    """
    try:
        if BIOMEDCLIP_MODEL is None or IMAGE_INDEX is None or not EMBEDDINGS or BIOMEDCLIP_PREPROCESS is None:
            return {"error": "Model, preprocess, or embeddings not loaded", "results": []}
        dicom = pydicom.dcmread(file.file)
        from PIL import Image
        img = dicom.pixel_array
        if img.ndim == 3:
            img = img[img.shape[0] // 2]
        img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
        pil_image = Image.fromarray(img)
        image_input = BIOMEDCLIP_PREPROCESS(pil_image).unsqueeze(0)
        with torch.no_grad():
            image_features = BIOMEDCLIP_MODEL.encode_image(image_input)
            embedding = image_features.squeeze().cpu().numpy().astype('float32')
        if hasattr(embedding, 'flatten'):
            embedding = embedding.flatten()
        else:
            embedding = embedding.reshape(-1)
        try:
            index_dim = IMAGE_INDEX.d
        except AttributeError:
            index_dim = getattr(IMAGE_INDEX, 'dim', None)
        if index_dim and embedding.shape[0] != index_dim:
            return {
                "error": f"Embedding dimension mismatch. Expected {index_dim}, got {embedding.shape[0]}",
                "results": []
            }
        _, indices = IMAGE_INDEX.search(np.array([embedding]), k=10)
        results = [EMBEDDINGS[i] for i in indices[0]]
        return {"results": results}
    except Exception as exc:
        return {"error": str(exc), "results": []}


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint for API and model status."""
    return {
        "status": "healthy",
        "model_loaded": BIOMEDCLIP_MODEL is not None,
        "embeddings_loaded": len(EMBEDDINGS) > 0
    }
