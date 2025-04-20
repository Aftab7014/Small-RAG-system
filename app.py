from flask import Flask, request, jsonify
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_groq import ChatGroq
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os
import redis

load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Global variables
vectordb = None
# qa_systems = {}
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
redis_client = redis.from_url(REDIS_URL)

prompt_template = """
You are an AI assistant helping answer questions based on context documents and chat history.

Use the following context and conversation history to answer the user's current question.

Context:
{context}

Chat History:
{chat_history}

Current Question:
{question}

Helpful Answer:
"""

PROMPT = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template=prompt_template,
)


# Initialize LLM
llm = ChatGroq(
    model="llama3-70b-8192",
    temperature=0,
    api_key=os.getenv("GROQ_API")
)

def load_documents(directory_path):
    """Load all PDF and text files from a directory"""
    if not os.path.exists(directory_path):
        raise ValueError(f"Directory {directory_path} does not exist")
    
    documents = []
    for file in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file)
        try:
            if file.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            elif file.endswith(".txt"):
                loader = TextLoader(file_path)
                documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading {file}: {str(e)}")
            continue
    
    if not documents:
        raise ValueError("No valid documents found in directory")
    return documents

def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(documents)

def get_hf_embeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    return HuggingFaceEmbeddings(model_name=model_name)


def create_vector_store(chunks, embedding):
    # Create and save FAISS index
    vectorstore = FAISS.from_documents(chunks, embedding)
    vectorstore.save_local("faiss_index")
    return vectorstore


def get_memory(session_id: str) -> ConversationBufferMemory:
    """Wire Redis history → BufferMemory with matching keys."""
    history = RedisChatMessageHistory(session_id=session_id, url=REDIS_URL)
    return ConversationBufferMemory(
        memory_key="chat_history",  # what the chain’s prompt expects
        input_key="question",       # what you’ll pass in
        output_key="answer",        # what the model returns
        chat_memory=history,
        return_messages=True,
    )



def get_qa_system(session_id: str) -> ConversationalRetrievalChain:
    memory = get_memory(session_id)
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": PROMPT},
        verbose=False,
    )



@app.route("/ask", methods=["POST"])
def ask_question():
    data       = request.json
    session_id = data.get("session_id", "default")
    query      = data.get("query", "").strip()
    if not query:
        return jsonify(error="No query provided"), 400

    qa     = get_qa_system(session_id)
    # CALL the chain normally, passing the same input_key you configured
    result = qa({"question": query})

    history = qa.memory.chat_memory.messages
    return jsonify({
        "answer":     result["answer"],
        "session_id": session_id,
        "history": [
            f"{i+1}. {m.type.title()}: {m.content}"
            for i, m in enumerate(history)
        ]
    })



# @app.route('/session', methods=['POST'])
# def validate_session():
#     session_id = request.json.get('session_id', 'default')
#     exists = session_id in qa_systems
#     return jsonify({
#         "session_id": session_id,
#         "exists": exists,
#         "message": "Session is maintained correctly" if exists else "New session created"
#     })

if __name__ == "__main__":
    # Configuration
    DOCUMENTS_DIR = "./documents"
    FAISS_INDEX = "faiss_index"
    
    try:
        # Initialize document processing
        print("Loading documents...")
        documents = load_documents(DOCUMENTS_DIR)
        
        print("Chunking documents...")
        chunks = chunk_documents(documents)
        
        print("Loading embeddings...")
        embedding = get_hf_embeddings()
        
        # Initialize vector store
        print("Creating vector store...")
        if os.path.exists(FAISS_INDEX):
            print("Loading existing FAISS index...")
            vectordb = FAISS.load_local(
                FAISS_INDEX,
                embedding,
                allow_dangerous_deserialization=True  # Required for loading
            )
        else:
            print("Creating new FAISS index...")
            vectordb = create_vector_store(chunks, embedding)

        # Start Flask app
        print("Starting API server...")
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    except Exception as e:
        print(f"Failed to initialize application: {str(e)}")