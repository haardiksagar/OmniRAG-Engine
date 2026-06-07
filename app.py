from src.search import RAGSearch
from src.vectorstore import FaissVectorStore
from src.data_loader import load_all_documents
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


# Example usage
if __name__ == "__main__":

    docs = load_all_documents("data")
    store = FaissVectorStore("faiss_store")
    store.build_from_documents(docs)
    store.load()
    print(store.query("What are biomaterials?", top_k=5))
    # If you want to increase the length and detail of the final response,
    # increasing top_k (e.g., to 5 or 10) is one way to do it
    rag_search = RAGSearch()
    query = "What are biomaterials?"
    summary = rag_search.search_and_summarize(query, top_k=5)
    print("Summary:", summary)



