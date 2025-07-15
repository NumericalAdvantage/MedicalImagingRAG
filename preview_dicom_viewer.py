"""Streamlit DICOM File Viewer for previewing DICOM metadata and images.

This app allows users to select and preview DICOM files from a directory.
"""

import os
from typing import Any, Dict

import pydicom
import streamlit as st
from pydicom import FileDataset


def get_dicom_metadata(ds: FileDataset) -> Dict[str, Any]:
    """Extract metadata from a DICOM dataset.

    Args:
        ds: DICOM dataset.
    Returns:
        Dictionary of DICOM metadata.
    """
    return {
        elem.keyword or str(elem.tag): str(elem.value)
        for elem in ds
        if elem.keyword and not elem.VR == "SQ"
    }


def main() -> None:
    """Main function to run the DICOM file viewer UI."""
    st.set_page_config(layout="wide")
    st.title("🧠 DICOM File Viewer")

    dicom_root = "data"
    if not os.path.exists(dicom_root):
        st.error(f"Directory '{dicom_root}' does not exist.")
        st.stop()

    dicom_files = [f for f in os.listdir(dicom_root) if f.lower().endswith(".dcm")]
    if not dicom_files:
        st.warning("No DICOM files found in the data directory.")
        st.stop()

    selected_file = st.selectbox("🩻 Select a DICOM File", dicom_files)
    if selected_file:
        file_path = os.path.join(dicom_root, selected_file)
        try:
            ds = pydicom.dcmread(file_path)
            st.subheader("📄 DICOM Metadata")
            st.json(get_dicom_metadata(ds))
            if "PixelData" in ds:
                st.subheader("🖼️ Image Preview")
                try:
                    img = ds.pixel_array
                    if img.ndim == 2:
                        st.image(img, caption=selected_file, clamp=True)
                    elif img.ndim == 3:
                        slice_idx = st.slider("Select Slice", 0, img.shape[0] - 1, 0)
                        st.image(img[slice_idx], caption=f"Slice {slice_idx}", clamp=True)
                    else:
                        st.warning("Unsupported image dimensions.")
                except Exception as exc:
                    st.error(f"Image render error: {exc}")
            else:
                st.info("This DICOM has no image data.")
        except Exception as exc:
            st.error(f"Could not load DICOM file: {exc}")


if __name__ == "__main__":
    main()
