import os
import numpy as np
import pickle
from typing import List, Any
from sentence_transformers import SentenceTransformer

try:
    import faiss
except ImportError:
    faiss = None

try:
    from .embedding import EmbeddingPipeline
except ImportError:
    from embedding import EmbeddingPipeline

class FaissVectorStore:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.index = None
        self.embeddings = None
        self.metadata = []
        self.embedding_model = embedding_model
        self.model = SentenceTransformer(embedding_model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_faiss = faiss is not None
        print(f"[INFO] Loaded embedding model: {embedding_model}")

    def build_from_documents(self, documents: List[Any]):
        print(f"[INFO] Building vector store from {len(documents)} raw documents...")
        emb_pipe = EmbeddingPipeline(model_name=self.embedding_model, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks = emb_pipe.chunk_documents(documents)
        embeddings = emb_pipe.embed_chunks(chunks)
        metadatas = [{"text": chunk.page_content} for chunk in chunks]
        self.add_embeddings(np.array(embeddings).astype('float32'), metadatas)
        self.save()
        print(f"[INFO] Vector store built and saved to {self.persist_dir}")

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any] = None):
        dim = embeddings.shape[1]
        if self.use_faiss and self.index is None:
            self.index = faiss.IndexFlatL2(dim)
        if self.use_faiss:
            self.index.add(embeddings)
        else:
            self.embeddings = embeddings if self.embeddings is None else np.vstack([self.embeddings, embeddings])
        if metadatas:
            self.metadata.extend(metadatas)
        index_name = "Faiss" if self.use_faiss else "NumPy"
        print(f"[INFO] Added {embeddings.shape[0]} vectors to {index_name} index.")

    def save(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        if self.use_faiss:
            faiss.write_index(self.index, faiss_path)
        else:
            emb_path = os.path.join(self.persist_dir, "embeddings.npy")
            np.save(emb_path, self.embeddings)
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"[INFO] Saved vector index and metadata to {self.persist_dir}")

    def load(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        if self.use_faiss and os.path.exists(faiss_path):
            self.index = faiss.read_index(faiss_path)
        else:
            emb_path = os.path.join(self.persist_dir, "embeddings.npy")
            self.embeddings = np.load(emb_path)
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)
        print(f"[INFO] Loaded vector index and metadata from {self.persist_dir}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        if self.use_faiss:
            D, I = self.index.search(query_embedding, top_k)
        else:
            if self.embeddings is None or len(self.embeddings) == 0:
                return []
            distances = np.linalg.norm(self.embeddings - query_embedding[0], axis=1)
            I = np.argsort(distances)[:top_k]
            D = distances[I]
        results = []
        for idx, dist in zip(I[0] if self.use_faiss else I, D[0] if self.use_faiss else D):
            meta = self.metadata[idx] if idx < len(self.metadata) else None
            results.append({"index": idx, "distance": dist, "metadata": meta})
        return results

    def query(self, query_text: str, top_k: int = 5):
        print(f"[INFO] Querying vector store for: '{query_text}'")
        query_emb = self.model.encode([query_text]).astype('float32')
        return self.search(query_emb, top_k=top_k)

# Example usage
if __name__ == "__main__":
    try:
        from .data_loader import load_all_documents
    except ImportError:
        from data_loader import load_all_documents
    docs = load_all_documents("data")
    store = FaissVectorStore("faiss_store")
    store.build_from_documents(docs)
    store.load()
    print(store.query("What is attention mechanism?", top_k=3))
