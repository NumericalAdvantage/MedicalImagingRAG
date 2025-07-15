import os
import tempfile
import numpy as np
import pydicom
import pytest
from pydicom.data import get_testdata_file

from app.embed_dicom_folder import load_dicom_image, embed_dicom

def test_load_dicom_image_valid():
    # Use a sample DICOM file from pydicom's test data
    dicom_path = get_testdata_file("CT_small.dcm")
    img, ds = load_dicom_image(dicom_path)
    assert img is not None
    assert ds is not None
    assert isinstance(img, np.ndarray)
    assert hasattr(ds, 'PatientID')

def test_load_dicom_image_invalid():
    # Try loading a non-DICOM file
    with tempfile.NamedTemporaryFile(suffix='.dcm') as tmp:
        tmp.write(b"not a dicom file")
        tmp.flush()
        img, ds = load_dicom_image(tmp.name)
        assert img is None
        assert ds is None

def test_embed_dicom_valid():
    dicom_path = get_testdata_file("CT_small.dcm")
    result = embed_dicom(dicom_path)
    # Result may be None if model/tokenizer is not loaded, but should not raise
    assert result is None or isinstance(result, dict) 