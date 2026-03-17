import os
import glob
import streamlit as st
from index import Indexing
from query import rag_query

SAVE_DIR = "uploaded_pdfs"

Vectors_created_and_save = False

os.makedirs(SAVE_DIR, exist_ok=True)

st.title("📄 PDF Uploader")

uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)


def Save_PDFs(uploaded_files):
    saved_files = []

    for file in uploaded_files:
        file_path = os.path.join(SAVE_DIR, file.name)

        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

        saved_files.append(file.name)


def DeletePDFS(Dir):
    for file in os.listdir(Dir):
        file_path = os.path.join(Dir, file)
        os.remove(file_path)


if st.button("Create Vectors"):
    Save_PDFs(uploaded_files)
    uploaded_files = glob.glob('./uploaded_pdfs/*.pdf')
    Vectors_created = Indexing(uploaded_files, './vectordb')
    DeletePDFS(SAVE_DIR)
    if Vectors_created:
        Vectors_created_and_save = True
        st.write('Created Successfully')
    else:
        st.write('Something went wrong, please check pdf size and other details')

question = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if True:
        if not question:
            st.warning("Please enter a question.")
        else:
            with st.spinner("Thinking..."):
                answer = rag_query(question, './vectordb')

            st.subheader("Answer:")
            st.write(answer)
    else:
        st.write('No DB')
