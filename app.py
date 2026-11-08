import os

import streamlit as st

from src.retriever import Retriever
from src.vector_store import load_vector_store
from src.rag_pipeline import RAGPipeline
from src.ingestion import ingest_document

from src.document_manager import (
    get_documents,
    get_document_storage_path,
    document_is_ingested
)


# --------------------------------
# Page configuration
# --------------------------------

st.set_page_config(
    page_title="RAG Research Assistant",
    page_icon="🔬",
    layout="wide"
)


# --------------------------------
# Session state
# --------------------------------

if "selected_document" not in st.session_state:
    st.session_state.selected_document = None


# --------------------------------
# Sidebar
# --------------------------------

with st.sidebar:

    st.title("🔬 RAG Assistant")

    st.caption(
        "AI-powered research workspace"
    )

    st.divider()


    # Navigation
    page = st.radio(
        "Navigation",
        [
            "🧠 Research Assistant",
            "📄 Document"
        ]
    )


    st.divider()


    # Document library
    st.subheader("📚 Document Library")


    documents = get_documents()


    if documents:

        selected_document = st.selectbox(
            "Select a document",
            documents
        )

        st.session_state.selected_document = (
            selected_document
        )

    else:

        st.caption(
            "No documents uploaded yet."
        )


    st.divider()


    # Upload new document
    st.subheader("➕ Add Document")

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        label_visibility="collapsed"
    )


# --------------------------------
# Process uploaded document
# --------------------------------

if uploaded_file:

    os.makedirs(
        "data",
        exist_ok=True
    )


    pdf_path = os.path.join(
        "data",
        uploaded_file.name
    )


    # Save PDF
    with open(
        pdf_path,
        "wb"
    ) as file:

        file.write(
            uploaded_file.getbuffer()
        )


    # Process only if not already indexed
    if not document_is_ingested(
        uploaded_file.name
    ):

        with st.spinner(
            "Processing document..."
        ):

            ingest_document(
                pdf_path
            )

        st.success(
            f"Added: {uploaded_file.name}"
        )

        st.session_state.selected_document = (
            uploaded_file.name
        )

        st.rerun()


    else:

        st.session_state.selected_document = (
            uploaded_file.name
        )


# --------------------------------
# Get selected document
# --------------------------------

selected_document = (
    st.session_state.selected_document
)


# --------------------------------
# RESEARCH ASSISTANT PAGE
# --------------------------------

if page == "🧠 Research Assistant":

    st.title(
        "🧠 Research Assistant"
    )

    st.write(
        "Ask questions about the selected "
        "research document."
    )


    if selected_document:

        storage_path = (
            get_document_storage_path(
                selected_document
            )
        )


        if document_is_ingested(
            selected_document
        ):

            # Load selected document index
            index, chunks = load_vector_store(
                storage_path
            )


            # Create retriever
            retriever = Retriever(
                index,
                chunks
            )


            # Create RAG pipeline
            pipeline = RAGPipeline(
                retriever
            )


            st.info(
                f"Researching: "
                f"**{selected_document}**"
            )


            # Question
            query = st.text_input(
                "Ask a question",
                placeholder=(
                    "e.g. What is the main "
                    "objective of this research?"
                )
            )


            # Ask
            if st.button(
                "Ask",
                type="primary"
            ):

                if not query.strip():

                    st.warning(
                        "Please enter a question."
                    )

                else:

                    with st.spinner(
                        "Searching the document..."
                    ):

                        answer, sources = (
                            pipeline.ask(
                                query,
                                k=3
                            )
                        )


                    # Answer
                    st.subheader(
                        "Answer"
                    )

                    st.write(
                        answer
                    )


                    # Sources
                    st.subheader(
                        "Sources"
                    )


                    for i, source in enumerate(
                        sources
                    ):

                        with st.expander(
                            f"📄 Result "
                            f"{i + 1} — "
                            f"Page "
                            f"{source['page']}"
                        ):

                            st.write(
                                source["text"]
                            )

                            st.caption(
                                f"Similarity "
                                f"distance: "
                                f"{source['distance']:.4f}"
                            )

        else:

            st.warning(
                "This document has not been "
                "processed yet."
            )


    else:

        st.info(
            "Upload or select a document "
            "from the sidebar to begin."
        )


# --------------------------------
# DOCUMENT PAGE
# --------------------------------

elif page == "📄 Document":

    st.title(
        "📄 Document"
    )


    if selected_document:

        pdf_path = os.path.join(
            "data",
            selected_document
        )


        if os.path.exists(
            pdf_path
        ):

            st.write(
                f"**File:** "
                f"{selected_document}"
            )


            # PDF information
            from pypdf import PdfReader

            reader = PdfReader(
                pdf_path
            )


            st.write(
                f"**Pages:** "
                f"{len(reader.pages)}"
            )


            st.divider()


            # PDF viewer
            try:

                st.pdf(
                    pdf_path,
                    height=900
                )

            except Exception:

                st.warning(
                    "The built-in PDF viewer "
                    "is unavailable. "
                    "You can open the PDF from "
                    "the data folder."
                )


        else:

            st.error(
                "The selected document "
                "could not be found."
            )


    else:

        st.info(
            "Select a document from the "
            "sidebar to view it."
        )