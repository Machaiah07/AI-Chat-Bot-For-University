Campus Assistant RAG
A lightweight Retrieval-Augmented Generation (RAG) campus assistant using Google Gemini + FAISS. Minimal, single-file README with only one icon.
Quick info
Files: main.py, rag.py, vector.py, requirements.txt
Purpose: answer campus queries (academic calendar, handbook, etc.) using semantic search + generative replies.
Setup:
git clone https://github.com/Machaiah07/Campus-Assistant-RAG.git
cd Campus-Assistant-RAG
python -m venv venv
# mac/linux
source venv/bin/activate
# windows
# venv\Scripts\activate
pip install -r requirements.txt
Config
Create a .env with:
GOOGLE_API_KEY=your_google_genai_api_key_here
Project layout
main.py     # entrypoint (CLI / interface)
rag.py      # RAG pipeline (retrieval + generation)
vector.py   # vector store + embeddings (FAISS)
requirements.txt
Author - Divin Machaiah KV
