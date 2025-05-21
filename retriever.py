from typing import List
import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder


from langchain_core.tools import tool

from langchain_community.vectorstores import FAISS
#from palimpsest import Palimpsest

import config

def load_vectorstore(file_path: str, embedding_model_name: str) -> FAISS:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No vectorstore found at {file_path}")

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

    return FAISS.load_local(
        file_path, embeddings, allow_dangerous_deserialization=True
    )


def get_retriever():
    #Load document store from persisted storage
    #loading list of problem numbers as ids
    
    vector_store_path = config.ASSISTANT_INDEX_FOLDER
    vectorstore = load_vectorstore(vector_store_path, config.EMBEDDING_MODEL)
    reranker_model = HuggingFaceCrossEncoder(model_name=config.RERANKING_MODEL)
    RERANKER = CrossEncoderReranker(model=reranker_model, top_n=2)
    MAX_RETRIEVALS = 5
    
    #with open(f'{vector_store_path}/docstore.pkl', 'rb') as file:
    #    documents = pickle.load(file)

    #doc_ids = [doc.metadata.get('problem_number', '') for doc in documents]
    #store = InMemoryByteStore()
    #id_key = "problem_number"
    #multi_retriever = MultiVectorRetriever(
    #        vectorstore=vectorstore,
    #        byte_store=store,
    #        id_key=id_key,
    #        search_kwargs={"k": MAX_RETRIEVALS},
    #    )
    #multi_retriever.docstore.mset(list(zip(doc_ids, documents)))
    retriever = ContextualCompressionRetriever(
            base_compressor=RERANKER, base_retriever=vectorstore.as_retriever(search_kwargs={"k": MAX_RETRIEVALS})
            )

    def search(query: str) -> List[Document]:
        result = retriever.invoke(query, search_kwargs={"k": 2})
        return result
    return search


search = get_retriever()

@tool
def search_kb(query: str) -> str:
    """Retrieves from knowledgebase context suitable for the query. Shall be always used when user asks question.
    Args:
        query: a query to knowledgebase which helps answer user's question
    Returns:
        Context from knowledgebase suitable for the query.
    """
    
    if found_docs := search(query):
        return "\n\n".join([doc.page_content for doc in found_docs[:30]])
    else:
        return "No matching information found."

if __name__ == '__main__':
    answer = search_kb("какие есть ЖК?")
    print(answer)



