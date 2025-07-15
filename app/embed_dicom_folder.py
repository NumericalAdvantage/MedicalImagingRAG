"""Embeds DICOM files using BioMedCLIP or DenseNet and writes to JSONL.

This script processes a folder of DICOM files, extracts embeddings and metadata,
and writes them to a JSONL file for use in search and retrieval.
"""

import os
import json
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pydicom
import torch
from monai.transforms.compose import Compose
from monai.transforms.spatial.array import Resize
from monai.transforms.intensity.array import ScaleIntensity


def load_biomedclip_models() -> Tuple[Any, Any, Any]:
    """Load BioMedCLIP model and tokenizer, fallback to DenseNet if needed.

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


def load_dicom_image(file_path: str) -> Tuple[Optional[np.ndarray], Optional[pydicom.dataset.FileDataset]]:
    """Load a DICOM file and return a 3-channel image and the dataset.

    Args:
        file_path: Path to the DICOM file.

    Returns:
        Tuple of (3-channel image as np.ndarray, DICOM dataset) or (None, None) on error.
    """
    try:
        ds = pydicom.dcmread(file_path)
        if 'PixelData' not in ds:
            raise ValueError("No PixelData in DICOM")
        img = ds.pixel_array
        if img.ndim == 3:
            img = img[img.shape[0] // 2]
        if img.ndim == 2:
            img = np.expand_dims(img, axis=0)  # (1, H, W)
        if img.ndim != 3:
            raise ValueError(f"Unsupported image dimensions: {img.shape}")
        img = img.astype(np.float32)
        # Convert to 3 channels for BioMedCLIP (H, W) -> (H, W, 3)
        img_3ch = np.stack([img[0]] * 3, axis=-1)
        return img_3ch, ds
    except Exception as exc:
        print(f"Failed to process {file_path}: {exc}")
        traceback.print_exc()
        return None, None


def embed_dicom(file_path: str) -> Optional[Dict[str, Any]]:
    """Embed a DICOM file using BioMedCLIP or fallback model.

    Args:
        file_path: Path to the DICOM file.

    Returns:
        Dict with embeddings and metadata, or None on error.
    """
    try:
        img_3ch, ds = load_dicom_image(file_path)
        if img_3ch is None or ds is None:
            return None
        text = f"{getattr(ds, 'Modality', '')}, {getattr(ds, 'StudyDescription', '')}, {getattr(ds, 'SeriesDescription', '')}"
        if biomedclip_tokenizer is not None:
            tokens = biomedclip_tokenizer([text], context_length=256)
            with torch.no_grad():
                text_features = biomedclip_model.encode_text(tokens)
                text_emb = text_features.squeeze().cpu().numpy()
        else:
            text_emb = None
        if biomedclip_preprocess is not None:
            from PIL import Image
            img_np = img_3ch.numpy() if hasattr(img_3ch, 'numpy') else img_3ch
            pil_image = Image.fromarray(img_np.astype(np.uint8))
            image_tensor = biomedclip_preprocess(pil_image).unsqueeze(0)
            with torch.no_grad():
                image_emb = biomedclip_model.encode_image(image_tensor).squeeze().cpu().numpy()
        else:
            if not isinstance(img_3ch, torch.Tensor):
                tensor = torch.as_tensor(img_3ch)
            else:
                tensor = img_3ch
            if tensor.ndim == 4 and tensor.shape[-1] == 3:
                tensor = tensor.permute(0, 3, 1, 2)
            elif tensor.ndim == 3 and tensor.shape[-1] == 3:
                tensor = tensor.permute(2, 0, 1).unsqueeze(0)
            if tensor.ndim == 4 and tensor.shape[1] != 1:
                tensor = tensor[:, 0:1, :, :]
            elif tensor.ndim == 3 and tensor.shape[0] != 1:
                tensor = tensor[0:1, :, :].unsqueeze(0)
            elif tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            if not callable(biomedclip_model):
                print(f"Error: biomedclip_model is not callable, got type {type(biomedclip_model)}")
                return None
            with torch.no_grad():
                image_emb = biomedclip_model(tensor).squeeze().numpy()
        return {
            "filename": str(file_path),
            "text_embedding": text_emb.tolist() if text_emb is not None else None,
            "image_embedding": image_emb.tolist(),
            "metadata_text": text,
            "patient_id": getattr(ds, "PatientID", None),
        }
    except Exception as exc:
        print(f"Failed to process {file_path}: {exc}")
        traceback.print_exc()
        return None


def process_folder(folder_path: str, output_path: str = "embeddings/embeddings.jsonl") -> None:
    """Process all DICOM files in a folder and write embeddings to JSONL.

    Args:
        folder_path: Path to folder containing DICOM files.
        output_path: Path to output JSONL file.
    """
    os.makedirs("embeddings", exist_ok=True)
    with open(output_path, "w") as f:
        for file in Path(folder_path).rglob("*.dcm"):
            result = embed_dicom(file)
            if result:
                f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    biomedclip_model, biomedclip_preprocess, biomedclip_tokenizer = load_biomedclip_models()
    if biomedclip_model is not None:
        biomedclip_model.eval()
        print("✅ Loaded BioMedCLIP model (OpenCLIP)")
    process_folder("data/")
