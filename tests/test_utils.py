import pytest
import pydicom
from pydicom.data import get_testdata_file
from ui.streamlit_app import display_dicom_image, display_dicom_metadata

class DummyStreamlit:
    def __init__(self):
        self.errors = []
        self.subheaders = []
        self.texts = []
    def error(self, msg):
        self.errors.append(msg)
    def subheader(self, msg):
        self.subheaders.append(msg)
    def columns(self, n):
        return [self, self]
    def text(self, msg):
        self.texts.append(msg)


def test_display_dicom_image_runs(monkeypatch):
    dicom_path = get_testdata_file("CT_small.dcm")
    ds = pydicom.dcmread(dicom_path)
    # Patch st.error to dummy
    import ui.streamlit_app as sapp
    monkeypatch.setattr(sapp.st, "error", lambda msg: None)
    fig = display_dicom_image(ds, "Test DICOM Image")
    assert fig is None or hasattr(fig, "savefig")


def test_display_dicom_metadata_runs(monkeypatch):
    dicom_path = get_testdata_file("CT_small.dcm")
    ds = pydicom.dcmread(dicom_path)
    dummy_st = DummyStreamlit()
    import ui.streamlit_app as sapp
    monkeypatch.setattr(sapp, "st", dummy_st)
    display_dicom_metadata(ds, "Test Metadata")
    assert "Test Metadata" in dummy_st.subheaders 