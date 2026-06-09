import os
import pdfplumber
from PIL import Image
import pytesseract

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


# --------------------
# 1. Load ALL files (PDF + Images)
# --------------------
documents = []

data_folder = "data"

for filename in os.listdir(data_folder):
    filepath = os.path.join(data_folder, filename)

    # --------------------
    # PDF FILES
    # --------------------
    if filename.lower().endswith(".pdf"):

        with pdfplumber.open(filepath) as pdf:
            for page_num, page in enumerate(pdf.pages):

                page_text = page.extract_text()

                if page_text:
                    documents.append(
                        Document(
                            page_content=page_text,
                            metadata={
                                "source": filename,
                                "page": page_num + 1,
                                "type": "pdf"
                            }
                        )
                    )

    # --------------------
    # IMAGE FILES (OCR)
    # --------------------
    elif filename.lower().endswith((".jpg", ".jpeg", ".png")):

        image = Image.open(filepath)
        text = pytesseract.image_to_string(image)

        if text.strip():
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": filename,
                        "type": "image"
                    }
                )
            )


print(f"\nTotal Documents Loaded: {len(documents)}")


# --------------------
# 2. Chunking
# --------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)

print(f"Total Chunks Created: {len(chunks)}")


# --------------------
# 3. Embeddings
# --------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------
# 4. Store in ChromaDB
# --------------------
db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma_db"
)

print("\n Vector DB created successfully!")