# Medical Imaging RAG

A modular, Dockerized Retrieval-Augmented Generation (RAG) system for medical imaging. This project enables semantic search and question-answering over DICOM files using state-of-the-art vision-language models and FAISS vector search, with a modern Streamlit UI and FastAPI backend.

---

## **Project Overview**

- **Purpose:**
  - Enable clinicians and researchers to search, explore, and query large collections of DICOM medical images using natural language and image similarity.
  - Combine deep learning embeddings (BioMedCLIP, DenseNet) with fast vector search (FAISS) and LLM-based answer generation (Ollama/llama2).

- **Key Features:**
  - Text-based semantic search over DICOM metadata and content.
  - Image-based similarity search for DICOM files.
  - Modern web UI (Streamlit) for interactive exploration.
  - Modular, reproducible, and containerized (Docker Compose).

---

## **Architecture**

```
[User] ⇄ [Streamlit UI] ⇄ [FastAPI Backend] ⇄ [FAISS Index, Embeddings, Models]
                                              ⇄ [Ollama LLM API]
```

- **UI:** Streamlit app for text/image queries, DICOM preview, and results display.
- **Backend:** FastAPI app for embedding, search, and LLM orchestration.
- **Embeddings:** Precomputed using BioMedCLIP (vision-language) or DenseNet (fallback).
- **Index:** FAISS for fast nearest-neighbor search over embeddings.
- **LLM:** Ollama (Llama2 or compatible) for answer generation.

---

## **Models Used**

- **BioMedCLIP:**
  - Vision-language model for medical images and text.
  - Used for both text and image embedding.
  - HuggingFace Hub: `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`
- **DenseNet121:**
  - Fallback vision model for chest X-ray images.
- **Sentence Transformers:**
  - For text embedding (BioBERT, etc.)
- **Ollama LLM (Llama2):**
  - For answer generation from retrieved context.
- **FAISS:**
  - For fast similarity search over embeddings.

---

## **Functionality**

- **Text Query:**
  - Enter a diagnostic or research question.
  - Backend retrieves top-k relevant DICOMs using semantic search.
  - LLM generates an answer using retrieved context.
  - UI displays answer, context, and DICOM previews.

- **Image Search:**
  - Upload a DICOM file.
  - Backend finds visually similar images using embedding and FAISS.
  - UI displays similar images, metadata, and previews.

- **DICOM Preview:**
  - View metadata and image slices for any DICOM in the results.

- **Health/Status:**
  - Sidebar shows API/model/embedding status and LLM warmup.

---

## **Setup Instructions**

### 1. **Clone the Repository**
```bash
git clone https://github.com/YOUR_USERNAME/medical-imaging-rag.git
cd medical-imaging-rag
```

### 2. **Install Python Dependencies (for local dev)**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For testing
```

### 3. **Prepare Data and Embeddings**
- Place your DICOM files in the `data/` directory.
- Run embedding:
  ```bash
  python app/embed_dicom_folder.py
  python app/build_faiss_indices.py
  ```

### 4. **Run with Docker Compose (Recommended)**
```bash
docker compose up --build
```
- UI: [http://localhost:8501](http://localhost:8501)
- API: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## **API Endpoints**

- `POST /rag_query` — Text-based search and answer generation
- `POST /image_search` — Image similarity search
- `GET /health` — API/model/embedding status

---

## **Testing**

- Run all unit tests:
  ```bash
  pytest
  ```
- Tests cover embedding, indexing, API endpoints, and utility functions.

---

## **Customization & Extensibility**
- Swap out models by changing environment variables or code in `api.py`/`embed_dicom_folder.py`.
- Add new endpoints or UI features as needed.
- All configuration is modular and containerized for easy deployment.

---

## **Security & Privacy**
- DICOM files and embeddings are not committed to version control (see `.gitignore`).
- Ensure PHI is handled according to your institution’s policies.

---

## **License**
MIT

---

## **Acknowledgements**
- [BioMedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Ollama](https://ollama.com/)
- [Streamlit](https://streamlit.io/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [pydicom](https://pydicom.github.io/) 