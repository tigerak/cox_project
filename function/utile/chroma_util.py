import chromadb

class ChromaDB:
    def __init__(self):
        client = chromadb.PersistentClient(path="./smart/data/chroma_db")

        # self.collection = client.get_or_create_collection(name="test_manual")
        self.collection = client.get_or_create_collection(name="consult_manual")

    def add_data(self, id, title, content, embedding):
        self.collection.upsert(
            documents=[title],
            embeddings=[embedding],
            metadatas=[{"content": content}],
            ids=[id]
        )