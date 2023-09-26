from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings import HuggingFaceEmbeddings
import pinecone, time

from config import API_KEY, ENV

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent

directory = 'files'

loader = DirectoryLoader(directory)

documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

texts = splitter.split_documents(documents)

print("Chunk size: " + str(len(texts)))

#print(texts[0].page_content)

embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

pinecone.init(
    api_key=API_KEY,
    environment=ENV
)

print("Pinecone initialized...")

index_name = "answerdocllama"

if index_name in pinecone.list_indexes():
    try:
        pinecone.delete_index(index_name)
        time.sleep(1)
        print("Previous " + index_name + " deleted...")
    except:
        print("Erroring deleting index...")

print("Creating index...")
pinecone.create_index(
        name = index_name,
        metric = 'cosine',
        dimension=384
)
print("Index created...")

time.sleep(1)

print("Uploading...")

try:
    db = Pinecone.from_documents(texts, embeddings, index_name=index_name)
    print("Uploaded: ")
    print(pinecone.describe_index(index_name))

except:
    print("Error uploading to index...")
