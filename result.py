from langchain.llms import LlamaCpp
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import pinecone
from langchain import PromptTemplate
from langchain.chains import RetrievalQA

from config import API_KEY, ENV

template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Context: {context}
Question: {question}
Only return the helpful answer below and nothing else.
Helpful answer:
"""

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = LlamaCpp(model_path = "./llama-2-7b-chat.Q5_K_M.gguf",
               temperature = 0,
               n_ctx=2048,
               top_p = 1,
               callback_manager=callback_manager,
               verbose=True,
               )

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

pinecone.init(
    api_key=API_KEY,
    environment=ENV
)

db = Pinecone.from_existing_index("answerdocllama", embeddings)

# def get_similar_docs(query, k=2, score= False):
#     if score:
#         similar_docs = db.similarity_search_with_score(query, k = k)
#     else:
#         similar_docs = db.similarity_search(query, k = k)
#     return similar_docs

# chain = load_qa_chain(llm, chain_type="stuff")

# def get_answer(query):
#   similar_docs = get_similar_docs(query)
#   answer = chain.run(input_documents=similar_docs, question=query)
#   return answer

retriever = db.as_retriever()
prompt = PromptTemplate(
    template=template,
    input_variables=['context', 'question'])

qa_llm = RetrievalQA.from_chain_type(llm=llm,
                                     chain_type= 'stuff',
                                     retriever=retriever,
                                     return_source_documents=True,
                                     callbacks = callback_manager,
                                     chain_type_kwargs={'prompt': prompt})

# prompt = "What is quadratic polynomial regression?"
# output = qa_llm({'query': prompt})
# print(output["result"])

while True:
    q = input(f"Q: ")
    if q != "quit":
        # a = get_answer(q)
        a = qa_llm({'query': q})
        print("A: " + a["result"] + '\n')
    else:
        print("Quiting...")
        break

