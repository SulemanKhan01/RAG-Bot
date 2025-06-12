from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import os
from langchain_community.vectorstores import FAISS


# print("Current working directory:", os.getcwd())
# print("Full data path:", DATA_PATH)
# print("Files in data1/:", os.listdir(DATA_PATH)) 
# 


from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data1")

def load_pdf_files(data):
    loader = DirectoryLoader(
                              data,
                              glob='*.pdf',
                              loader_cls=PyPDFLoader)
    
    documents = loader.load()
    return documents

documents = load_pdf_files(data=DATA_PATH)
# print("Length of PDF documents:", len(documents))




# Step 2: Create Chunks
def create_chunks(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,
                                                 chunk_overlap=50)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks


text_chunks = create_chunks(documents)
# print("Length of PDF chunks:", len(text_chunks))


# Step 3: Embeddings
def get_embedding_model():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

embedding_model=get_embedding_model()


# Step 4: Store embeddings in FAISS
FAISS_Path = "vectorstores/db_faiss"
db = FAISS.from_documents(text_chunks , embedding_model)
db.save_local(FAISS_Path)