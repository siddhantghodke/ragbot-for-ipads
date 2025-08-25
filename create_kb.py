import os 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()

def load_data():
    topics = [
                "ipad_wikipedia_content.txt",
                "ipad_air_wikipedia_content.txt", 
                "ipad_general_wikipedia_content.txt",
                "ipad_mini_wikipedia_content.txt",
                "ipad_pro_wikipedia_content.txt"
            ]

    all_documents = []
    
    for topic in topics:
        if os.path.exists(topic):
            try:
                loader = TextLoader(topic, encoding='utf-8')
                documents = loader.load()
                all_documents.extend(documents)
                print(f"✅ Loaded: {topic}")
            except Exception as e:
                print(f"❌ Error loading {topic}: {e}")
        else:
            print(f"⚠️ File not found: {topic}")
    
    if not all_documents:
        raise ValueError("No documents were loaded successfully!")
    
    print(f"📚 Total documents loaded: {len(all_documents)}")
    return all_documents

def create_knowledgebase():
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        raise ValueError("❌ GOOGLE_API_KEY environment variable not found! Please set it in your environment.")
    
    print(f"🔑 API Key found: {api_key[:10]}...")
    
    data = load_data()
    text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,  
                    chunk_overlap=300,  
                    length_function=len,
                    separators=["\n\n", "\n", ". ", " ", ""]
                )
    split_docs = text_splitter.split_documents(data)

     # Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
                        model="models/embedding-001",
                        google_api_key=api_key
                    )

    # Create and save FAISS vector store
    vector_store = FAISS.from_documents(split_docs, embeddings)
    vector_store.save_local("vectorstore.db")
    print("Knowledge base saved to 'vectorstore.db'")

if __name__ == "__main__":
    create_knowledgebase()