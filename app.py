"""
StudyMate AI - Personalized Learning Content Creator
Main Streamlit Application Entry Point

A generative AI system that combines Retrieval-Augmented Generation (RAG),
Prompt Engineering, and Synthetic Data Generation to help students learn
from their study materials.

Author: [Your Name]
Course: Generative AI - Final Project
"""

import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv

from src.rag_engine import RAGEngine
from src.prompt_templates import PromptManager
from src.synthetic_data import SyntheticDataGenerator
from src.utils import load_document, chunk_text, validate_file

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="StudyMate AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -------------------- Session State Initialization --------------------
def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        "rag_engine": None,
        "prompt_manager": PromptManager(),
        "syn_generator": None,
        "uploaded_docs": [],
        "chat_history": [],
        "current_kb": "default",
        "model_choice": "claude-sonnet-4-5",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# -------------------- Sidebar: Configuration --------------------
def render_sidebar():
    """Render the sidebar with configuration and document upload."""
    with st.sidebar:
        st.title("📚 StudyMate AI")
        st.markdown("*Your AI-powered study companion*")
        st.divider()

        # Model selection
        st.subheader("⚙️ Configuration")
        model_choice = st.selectbox(
            "Select LLM Provider",
            options=[
                "claude-sonnet-4-5",
                "claude-haiku-4-5",
                "gpt-4o-mini",
                "local-llama",
            ],
            index=0,
            help="Choose the language model backend.",
        )
        st.session_state.model_choice = model_choice

        # Chunking parameters
        with st.expander("🔧 Advanced RAG Settings"):
            chunk_size = st.slider("Chunk Size (tokens)", 200, 1500, 512, 50)
            chunk_overlap = st.slider("Chunk Overlap", 0, 200, 50, 10)
            top_k = st.slider("Top-K Retrieval", 1, 10, 4, 1)
            st.session_state.chunk_size = chunk_size
            st.session_state.chunk_overlap = chunk_overlap
            st.session_state.top_k = top_k

        st.divider()

        # Document upload
        st.subheader("📄 Upload Study Materials")
        uploaded_files = st.file_uploader(
            "Drop PDF, TXT, or MD files here",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
        )

        if uploaded_files and st.button("🔨 Build Knowledge Base", type="primary"):
            with st.spinner("Processing documents and building vector index..."):
                build_knowledge_base(uploaded_files)

        if st.session_state.uploaded_docs:
            st.success(f"✅ {len(st.session_state.uploaded_docs)} document(s) indexed")
            for doc in st.session_state.uploaded_docs:
                st.caption(f"• {doc}")

        st.divider()
        st.caption(
            "Built with LangChain, ChromaDB, Sentence-Transformers, and Streamlit."
        )


def build_knowledge_base(uploaded_files):
    """Process uploaded files and build the RAG vector store."""
    # Initialize RAG engine if needed
    if st.session_state.rag_engine is None:
        st.session_state.rag_engine = RAGEngine(
            collection_name=st.session_state.current_kb,
            chunk_size=st.session_state.get("chunk_size", 512),
            chunk_overlap=st.session_state.get("chunk_overlap", 50),
        )

    docs_processed = []
    for uploaded_file in uploaded_files:
        if not validate_file(uploaded_file):
            st.warning(f"⚠️ Skipped invalid file: {uploaded_file.name}")
            continue
        try:
            text = load_document(uploaded_file)
            st.session_state.rag_engine.add_document(
                text=text, metadata={"source": uploaded_file.name}
            )
            docs_processed.append(uploaded_file.name)
        except Exception as e:
            st.error(f"Failed to process {uploaded_file.name}: {e}")

    st.session_state.uploaded_docs = docs_processed
    # Initialize synthetic data generator tied to current KB
    st.session_state.syn_generator = SyntheticDataGenerator(
        rag_engine=st.session_state.rag_engine,
        prompt_manager=st.session_state.prompt_manager,
        model=st.session_state.model_choice,
    )


# -------------------- Main Tabs --------------------
def render_main_area():
    """Render the main tabbed interface."""
    st.title("StudyMate AI — Learn Smarter")
    st.markdown(
        "Ask questions, generate quizzes, create flashcards, and summarize your materials."
    )

    if st.session_state.rag_engine is None or not st.session_state.uploaded_docs:
        st.info(
            "👈 Upload study materials in the sidebar and click **Build Knowledge Base** to get started."
        )
        render_welcome_screen()
        return

    tab_chat, tab_quiz, tab_flash, tab_summary, tab_explain = st.tabs(
        ["💬 Q&A Chat", "📝 Quiz Generator", "🎴 Flashcards", "📋 Summarizer", "🧠 Explain Like I'm 5"]
    )

    with tab_chat:
        render_qa_tab()
    with tab_quiz:
        render_quiz_tab()
    with tab_flash:
        render_flashcard_tab()
    with tab_summary:
        render_summary_tab()
    with tab_explain:
        render_explain_tab()


def render_welcome_screen():
    """Show a welcome screen when no documents are loaded."""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📚 RAG-Powered")
        st.write(
            "Ask questions grounded in your own notes and textbooks. "
            "No hallucinations from the open web."
        )
    with col2:
        st.markdown("### 🎯 Prompt-Engineered")
        st.write(
            "Carefully designed prompt templates for each learning task, "
            "with structured outputs and error handling."
        )
    with col3:
        st.markdown("### ⚡ Synthetic Quizzes")
        st.write(
            "Automatically generate practice questions, flashcards, "
            "and summaries from your materials."
        )


def render_qa_tab():
    """Question-answering chat interface with RAG."""
    st.subheader("Ask anything about your study materials")

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📖 Sources used"):
                    for src in msg["sources"]:
                        st.caption(f"• {src}")

    # Chat input
    if query := st.chat_input("Type your question..."):
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving context and generating answer..."):
                answer, sources = answer_question(query)
            st.markdown(answer)
            if sources:
                with st.expander("📖 Sources used"):
                    for src in sources:
                        st.caption(f"• {src}")

        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )


def answer_question(query: str):
    """Run the RAG pipeline to answer a question."""
    rag = st.session_state.rag_engine
    pm = st.session_state.prompt_manager

    # Retrieve relevant chunks
    retrieved = rag.query(query, top_k=st.session_state.get("top_k", 4))
    context = "\n\n".join([r["text"] for r in retrieved])
    sources = list({r["metadata"].get("source", "unknown") for r in retrieved})

    # Build prompt and call LLM
    prompt = pm.build_qa_prompt(query=query, context=context)
    answer = pm.call_llm(prompt, model=st.session_state.model_choice)
    return answer, sources


def render_quiz_tab():
    """Quiz generation tab."""
    st.subheader("Generate practice quizzes")
    col1, col2, col3 = st.columns(3)
    with col1:
        num_questions = st.number_input("Number of questions", 1, 20, 5)
    with col2:
        q_type = st.selectbox("Question type", ["MCQ", "Short Answer", "True/False", "Mixed"])
    with col3:
        difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"])

    topic = st.text_input(
        "Topic (optional — leave blank for full-document quiz)",
        placeholder="e.g., 'photosynthesis' or 'Chapter 3'",
    )

    if st.button("🎯 Generate Quiz", type="primary"):
        with st.spinner("Synthesizing questions..."):
            quiz = st.session_state.syn_generator.generate_quiz(
                topic=topic if topic else None,
                num_questions=num_questions,
                question_type=q_type,
                difficulty=difficulty,
            )
        display_quiz(quiz)


def display_quiz(quiz):
    """Render a generated quiz nicely."""
    if not quiz:
        st.error("Quiz generation failed. Please try again.")
        return
    for i, q in enumerate(quiz, 1):
        with st.container(border=True):
            st.markdown(f"**Q{i}. {q.get('question', '')}**")
            if q.get("options"):
                for opt_letter, opt_text in q["options"].items():
                    st.markdown(f"&nbsp;&nbsp;**{opt_letter})** {opt_text}")
            with st.expander("✅ Show answer and explanation"):
                st.markdown(f"**Answer:** {q.get('answer', 'N/A')}")
                if q.get("explanation"):
                    st.markdown(f"**Explanation:** {q['explanation']}")


def render_flashcard_tab():
    """Flashcard generation tab."""
    st.subheader("Create flashcards from your notes")
    num_cards = st.slider("Number of flashcards", 5, 30, 10)
    topic = st.text_input("Topic (optional)", key="flash_topic")

    if st.button("🎴 Generate Flashcards", type="primary"):
        with st.spinner("Creating flashcards..."):
            cards = st.session_state.syn_generator.generate_flashcards(
                topic=topic if topic else None, num_cards=num_cards
            )
        display_flashcards(cards)


def display_flashcards(cards):
    """Display flashcards in a grid."""
    if not cards:
        st.error("Flashcard generation failed.")
        return
    cols_per_row = 2
    for i in range(0, len(cards), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            if i + j < len(cards):
                card = cards[i + j]
                with col:
                    with st.container(border=True):
                        st.markdown(f"**Front:** {card.get('front', '')}")
                        with st.expander("Show back"):
                            st.markdown(card.get("back", ""))


def render_summary_tab():
    """Summary generation tab."""
    st.subheader("Summarize your materials")
    summary_length = st.selectbox(
        "Summary length",
        ["One-paragraph", "Bullet points (5-10)", "Detailed outline", "Executive summary"],
    )
    topic = st.text_input("Focus topic (optional)", key="sum_topic")

    if st.button("📋 Generate Summary", type="primary"):
        with st.spinner("Summarizing..."):
            summary = st.session_state.syn_generator.generate_summary(
                topic=topic if topic else None, length=summary_length
            )
        st.markdown(summary)


def render_explain_tab():
    """Explain-like-I'm-5 tab."""
    st.subheader("Get simple explanations of complex topics")
    concept = st.text_input("What concept would you like explained?")
    audience = st.selectbox(
        "Explain for a...",
        ["5-year-old", "middle-school student", "high-school student", "college freshman"],
    )
    if st.button("🧠 Explain It", type="primary") and concept:
        with st.spinner("Crafting explanation..."):
            explanation = st.session_state.syn_generator.generate_eli5(
                concept=concept, audience=audience
            )
        st.markdown(explanation)


# -------------------- Main --------------------
def main():
    init_session_state()
    render_sidebar()
    render_main_area()


if __name__ == "__main__":
    main()
