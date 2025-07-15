"""Streamlit UI for Medical Imaging RAG system.

Provides a web interface for text and image search over DICOM data.
"""

import io
import json
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pydicom
import requests
import streamlit as st
from PIL import Image
from matplotlib.figure import Figure


def display_dicom_image(dicom_data: Any, title: str = "DICOM Image") -> Optional[Figure]:
    """Display a DICOM image using matplotlib.

    Args:
        dicom_data: DICOM dataset or numpy array.
        title: Title for the image.

    Returns:
        Matplotlib figure or None on error.
    """
    try:
        if hasattr(dicom_data, 'pixel_array'):
            pixel_array = dicom_data.pixel_array
        else:
            pixel_array = dicom_data
        if pixel_array.ndim == 3:
            pixel_array = pixel_array[pixel_array.shape[0] // 2]
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(pixel_array, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
        return fig
    except Exception as exc:
        st.error(f"Error displaying DICOM image: {exc}")
        return None


def display_dicom_metadata(dicom_data: Any, title: str = "DICOM Metadata") -> None:
    """Display DICOM metadata in a formatted way in Streamlit.

    Args:
        dicom_data: DICOM dataset.
        title: Section title.
    """
    st.subheader(title)
    metadata = {
        "Patient ID": getattr(dicom_data, 'PatientID', 'N/A'),
        "Modality": getattr(dicom_data, 'Modality', 'N/A'),
        "Study Date": getattr(dicom_data, 'StudyDate', 'N/A'),
        "Study Description": getattr(dicom_data, 'StudyDescription', 'N/A'),
        "Series Description": getattr(dicom_data, 'SeriesDescription', 'N/A'),
        "Image Size": f"{getattr(dicom_data, 'Rows', 'N/A')} x {getattr(dicom_data, 'Columns', 'N/A')}",
        "Bits Allocated": getattr(dicom_data, 'BitsAllocated', 'N/A'),
        "Pixel Spacing": getattr(dicom_data, 'PixelSpacing', 'N/A'),
    }
    col1, col2 = st.columns(2)
    for i, (key, value) in enumerate(metadata.items()):
        if i % 2 == 0:
            col1.text(f"**{key}:** {value}")
        else:
            col2.text(f"**{key}:** {value}")


def main() -> None:
    """Main function to run the Streamlit UI."""
    st.set_page_config(page_title="Medical Imaging RAG", layout="wide")
    st.title("🧠 Medical Image Search + RAG")

    tab1, tab2 = st.tabs(["Text Query", "Image Upload"])

    with tab1:
        query = st.text_input("Enter your diagnostic question")
        if st.button("Ask"):
            try:
                res = requests.post(
                    "http://rag-api:8000/rag_query",
                    json={"query": query},
                    timeout=300
                )
                if res.status_code == 200:
                    data = res.json()
                    if "error" in data:
                        st.error(f"Error: {data['error']}")
                    else:
                        st.write("### Context Results:")
                        context_items = data.get("context_items", [])
                        for i, item in enumerate(context_items):
                            st.write(f"**Result {i+1}:**")
                            st.code(item.get("metadata_text", ""))
                            filename = item.get("filename", "")
                            if filename and filename.endswith('.dcm'):
                                try:
                                    if 'data/' in filename:
                                        relative_path = filename.split('data/')[-1]
                                        full_path = f"/app/data/{relative_path}"
                                        with open(full_path, 'rb') as f:
                                            dicom_bytes = f.read()
                                        dicom_file = io.BytesIO(dicom_bytes)
                                        dicom_data = pydicom.dcmread(dicom_file)
                                        pixel_array = dicom_data.pixel_array
                                        if pixel_array.ndim == 3:
                                            pixel_array = pixel_array[pixel_array.shape[0] // 2]
                                        arr = pixel_array.astype(np.float32)
                                        arr -= arr.min()
                                        arr /= (arr.max() + 1e-5)
                                        arr = (arr * 255).astype(np.uint8)
                                        img = Image.fromarray(arr)
                                        img = img.resize((128, 128))
                                        st.image(img, caption=f"Result {i+1}", use_container_width=False)
                                        with st.expander("Show full image and metadata"):
                                            display_dicom_metadata(dicom_data, f"Result {i+1} Metadata")
                                            fig = display_dicom_image(dicom_data, f"Context Image {i+1}")
                                            if fig:
                                                st.pyplot(fig)
                                                plt.close(fig)
                                except Exception as exc:
                                    st.warning(f"Could not display DICOM preview for result {i+1}: {exc}")
                        st.write("### Answer:")
                        st.success(data.get("answer", ""))
                else:
                    st.error(f"API returned status code {res.status_code}")
            except Exception as exc:
                st.error(f"Error: {exc}")

    with tab2:
        st.write("### Upload DICOM File for Similar Image Search")
        uploaded_file = st.file_uploader("Choose a DICOM file", type=['dcm'])
        if uploaded_file is not None:
            st.write("### 📁 Uploaded DICOM File")
            try:
                dicom_data = pydicom.dcmread(uploaded_file)
                display_dicom_metadata(dicom_data, "Uploaded File Metadata")
                st.write("### 🖼️ Uploaded Image")
                fig = display_dicom_image(dicom_data, "Uploaded DICOM Image")
                if fig:
                    st.pyplot(fig)
                    plt.close(fig)
                if st.button("🔍 Search Similar Images"):
                    with st.spinner("Searching for similar images..."):
                        try:
                            uploaded_file.seek(0)
                            res = requests.post(
                                "http://rag-api:8000/image_search",
                                files={"file": uploaded_file},
                                timeout=300
                            )
                            if res.status_code == 200:
                                data = res.json()
                                if "error" in data:
                                    st.error(f"Error: {data['error']}")
                                else:
                                    st.write("### 🔍 Search Results")
                                    st.success(f"Found {len(data.get('results', []))} similar images")
                                    for i, result in enumerate(data.get("results", [])):
                                        st.write("---")
                                        col_img, col_meta = st.columns([1, 3])
                                        with col_img:
                                            try:
                                                filename = result.get("filename", "")
                                                if filename and filename.endswith('.dcm'):
                                                    if 'data/' in filename:
                                                        relative_path = filename.split('data/')[-1]
                                                        full_path = f"/app/data/{relative_path}"
                                                        with open(full_path, 'rb') as f:
                                                            dicom_bytes = f.read()
                                                        dicom_file = io.BytesIO(dicom_bytes)
                                                        result_dicom = pydicom.dcmread(dicom_file)
                                                        if hasattr(result_dicom, 'pixel_array'):
                                                            pixel_array = result_dicom.pixel_array
                                                            if pixel_array.ndim == 3:
                                                                pixel_array = pixel_array[pixel_array.shape[0] // 2]
                                                            arr = pixel_array.astype(np.float32)
                                                            arr -= arr.min()
                                                            arr /= (arr.max() + 1e-5)
                                                            arr = (arr * 255).astype(np.uint8)
                                                            img = Image.fromarray(arr)
                                                            img = img.resize((128, 128))
                                                            st.image(img, caption=f"Result {i+1}", use_container_width=False)
                                            except Exception as exc:
                                                st.warning(f"Could not display thumbnail for result {i+1}: {exc}")
                                        with col_meta:
                                            st.markdown(f"**Result {i+1}**")
                                            st.code(result.get("metadata_text", ""))
                                            with st.expander("Show full image and metadata"):
                                                try:
                                                    if 'result_dicom' in locals():
                                                        display_dicom_metadata(result_dicom, f"Result {i+1} Metadata")
                                                        fig = display_dicom_image(result_dicom, f"Similar Image {i+1}")
                                                        if fig:
                                                            st.pyplot(fig)
                                                            plt.close(fig)
                                                except Exception as exc:
                                                    st.error(f"Could not load DICOM file: {exc}")
                            else:
                                st.error(f"API returned status code {res.status_code}")
                        except requests.exceptions.RequestException as exc:
                            st.error(f"Failed to connect to API: {exc}")
                        except json.JSONDecodeError as exc:
                            st.error(f"Invalid response from API: {exc}")
                        except Exception as exc:
                            st.error(f"Unexpected error: {exc}")
            except Exception as exc:
                st.error(f"Error reading DICOM file: {exc}")

    with st.sidebar:
        st.header("System Status")
        try:
            health_res = requests.get("http://rag-api:8000/health", timeout=5)
            if health_res.status_code == 200:
                health_data = health_res.json()
                st.success("✅ API is healthy")
                st.info(f"Model loaded: {health_data.get('model_loaded', False)}")
                st.info(f"Embeddings loaded: {health_data.get('embeddings_loaded', False)}")
            else:
                st.error("❌ API health check failed")
        except Exception:
            st.error("❌ Cannot connect to API")
        st.markdown("---")
        st.subheader("Ollama Model Status")
        if 'ollama_warmup_status' not in st.session_state:
            with st.spinner("Warming up Ollama model (first request may take up to 2 minutes)..."):
                try:
                    warmup_res = requests.post(
                        "http://rag-api:8000/rag_query",
                        json={"query": "Say hello."},
                        timeout=300
                    )
                    if warmup_res.status_code == 200:
                        data = warmup_res.json()
                        if 'answer' in data and data['answer']:
                            st.session_state['ollama_warmup_status'] = 'success'
                            st.success("✅ Ollama model is ready.")
                        else:
                            st.session_state['ollama_warmup_status'] = 'fail'
                            st.error("❌ Ollama responded but no answer was returned.")
                    else:
                        st.session_state['ollama_warmup_status'] = 'fail'
                        st.error(f"❌ Ollama warm-up failed (status {warmup_res.status_code})")
                except requests.exceptions.Timeout:
                    st.session_state['ollama_warmup_status'] = 'fail'
                    st.error("❌ Ollama model warm-up timed out. It may still be loading. Please try again in a minute.")
                except Exception as exc:
                    st.session_state['ollama_warmup_status'] = 'fail'
                    st.error(f"❌ Ollama warm-up error: {exc}")
        else:
            if st.session_state['ollama_warmup_status'] == 'success':
                st.success("✅ Ollama model is ready.")
            else:
                st.error("❌ Ollama model is not ready. Try refreshing or check logs.")


if __name__ == "__main__":
    main()
