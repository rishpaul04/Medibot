🩺 Medibot: AI-Powered Medical Assistant (RAG)
Medibot is a Generative AI-powered chatbot designed to provide accurate, context-aware medical information. It utilizes Retrieval-Augmented Generation (RAG) to answer user queries based on verified medical documentation rather than relying solely on model training data, ensuring higher accuracy and reduced hallucinations.

The system features secure user authentication, fast inference via Groq, and a vector search backend for retrieving relevant medical context.

<img width="1919" height="1019" alt="Screenshot 2026-01-17 141110" src="https://github.com/user-attachments/assets/49240f62-93c4-4d06-b1f1-269e0980eff7" />

🚀 Key Features
Retrieval-Augmented Generation (RAG): Fetches relevant medical context from a Pinecone vector database before generating answers.

High-Speed Inference: Powered by Llama-3 (via Groq API) for near-instantaneous responses.

Source Citation: Provides references to the medical texts used to generate the answer (as seen in the interface).

Secure Authentication: Integrated Clerk Authentication for secure sign-up, login, and user management.

Memory & Context: Uses LangChain to orchestrate the retrieval and generation pipeline.

Interactive UI: Clean, responsive chat interface built with Bootstrap, HTML, and JS.

🛠️ Tech Stack
Backend: Python, Flask

LLM Engine: Groq (Llama-3.3-70b-versatile)

Vector Database: Pinecone

Orchestration: LangChain

Embeddings: HuggingFace (sentence-transformers/all-MiniLM-L6-v2)

Frontend: HTML5, CSS3, JavaScript, Bootstrap

⚙️ How It Works
Ingestion: Medical PDFs/Documents are embedded using HuggingFaceEmbeddings and stored in Pinecone.

Retrieval: When a user asks a question (e.g., "What is diabetes?"), the system searches Pinecone for the most similar document chunks.

Generation: The retrieved chunks + the user query are sent to Groq (Llama-3) via LangChain.

Response: The LLM generates a factual answer based only on the provided context, often citing the specific medical reference.

📦 Installation
Clone the repository:

Bash

git clone https://github.com/yourusername/medibot.git
cd medibot
Install dependencies:

Bash

pip install -r requirements.txt
Set up Environment Variables: Create a .env file and add your API keys:

Code snippet

PINECONE_API_KEY=your_pinecone_key
GROQ_API_KEY=your_groq_key
CLERK_PEM_PUBLIC_KEY=your_clerk_pem_key
VITE_CLERK_PUBLISHABLE_KEY=your_clerk_pub_key
Run the Application:

Bash

python app.py
🔮 Future Improvements
Implement chat history persistence (conversational memory).

Add support for voice input/output.

Expand the knowledge base to include recent medical journals.
