import os
import streamlit as st
import numpy as np
import pydicom
import matplotlib.pyplot as plt
from glob import glob

st.set_page_config(layout="wide")
DICOM_ROOT = "data"

def load_dicom_slices(folder_path):
    dicom_files = sorted(glob(os.path.join(folder_path, "*.dcm")))
    slices = []
    metadata = []
    for file in dicom_files:
        try:
            ds = pydicom.dcmread(file)
            slices.append(ds.pixel_array)
            metadata.append(ds)
        except Exception as e:
            st.warning(f"Could not read {file}: {e}")
    return slices, metadata

# Sidebar
st.sidebar.title(" Select DICOM Volume")
dicom_folders = [f for f in os.listdir(DICOM_ROOT) if os.path.isdir(os.path.join(DICOM_ROOT, f))]
selected_folder = st.sidebar.selectbox("Choose a folder:", dicom_folders)

# Load volume
folder_path = os.path.join(DICOM_ROOT, selected_folder)
slices, metadata = load_dicom_slices(folder_path)

# Show metadata
if metadata:
    ds = metadata[0]
    st.sidebar.markdown("###  DICOM Metadata")
    st.sidebar.text(f"Patient ID: {getattr(ds, 'PatientID', 'N/A')}")
    st.sidebar.text(f"Modality: {getattr(ds, 'Modality', 'N/A')}")
    st.sidebar.text(f"Study Date: {getattr(ds, 'StudyDate', 'N/A')}")
    st.sidebar.text(f"Series Description: {getattr(ds, 'SeriesDescription', 'N/A')}")

# Image Viewer
st.title(f" Viewing: `{selected_folder}`")
if slices:
    slice_idx = st.slider("Select Slice", 0, len(slices) - 1, 0)
    fig, ax = plt.subplots()
    ax.imshow(slices[slice_idx], cmap="gray")
    ax.axis("off")
    st.pyplot(fig)
else:
    st.warning("No DICOM slices found in this folder.")

# Embed trigger
if st.button(" Start Embedding This Volume"):
    os.system(f"python embed_dicom_folder.py")
    st.success("Embedding triggered.")
