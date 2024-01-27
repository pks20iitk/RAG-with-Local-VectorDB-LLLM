import uuid
from typing import List, Dict
from weaviate import Client
from weaviate.util import get_valid_uuid
from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import DataSourceMetadata
from unstructured.partition.json import partition_json
from sentence_transformers import SentenceTransformer
from data_ingest_pipeline.json import get_result_files


class WeaviateHandler:
    def __init__(self, db_url: str):
        self.client = Client(url=db_url)

    def create_schema(self, vectorizer: str = "none"):
        return {
            "classes": [
                {
                    "class": "Doc",
                    "description": "A generic document class",
                    "vectorizer": vectorizer,
                    "properties": [
                        {"name": "last_modified", "dataType": ["text"],
                         "description": "Last modified date for the document"},
                        {"name": "player", "dataType": ["text"], "description": "Player related to the document"},
                        {"name": "position", "dataType": ["text"],
                         "description": "Player Position related to the document"},
                        {"name": "text", "dataType": ["text"], "description": "Text content for the document"},
                    ],
                },
            ],
        }

    def upload_schema(self, my_schema):
        self.client.schema.delete_all()
        self.client.schema.create(my_schema)

    def count_documents(self) -> Dict:
        response = (
            self.client.query
            .aggregate("Doc")
            .with_meta_count()
            .do()
        )
        count = response
        return count


from typing import List


class DataProcessor:
    @staticmethod
    def compute_embedding(chunk_text: List[str], embedding_model, device):
        embeddings = embedding_model.encode(chunk_text, device=device)
        return embeddings

    @staticmethod
    def get_chunks_with_embeddings(elements, chunk_under_n_chars=500, chunk_new_after_n_chars=1500):
        for element in elements:
            if not isinstance(element.metadata.data_source, DataSourceMetadata):
                delattr(element.metadata, "data_source")

            if hasattr(element.metadata, "coordinates"):
                delattr(element.metadata, "coordinates")

        chunks = chunk_by_title(
            elements,
            combine_under_n_chars=chunk_under_n_chars,
            new_after_n_chars=chunk_new_after_n_chars
        )

        for i in range(len(chunks)):
            chunks[i] = {
                "last_modified": chunks[i].metadata.last_modified,
                "text": chunks[i].text,
            }

        chunk_texts = [x['text'] for x in chunks]
        embeddings = DataProcessor.compute_embedding(chunk_texts, embedding_model, device='cpu')
        return chunks, embeddings


class WeaviateDataUploader:
    def __init__(self, client):
        self.client = client

    def add_data_to_weaviate(self, files, chunk_under_n_chars=500, chunk_new_after_n_chars=1500):
        for filename in files:
            try:
                elements = partition_json(filename=filename)
                chunks, embeddings = DataProcessor.get_chunks_with_embeddings(elements, chunk_under_n_chars, chunk_new_after_n_chars)
            except IndexError as e:
                print(e)
                continue

            print(f"Uploading {len(chunks)} chunks for {str(filename)}.")
            for i, chunk in enumerate(chunks):
                self.client.batch.add_data_object(
                    data_object=chunk,
                    class_name="doc",
                    uuid=get_valid_uuid(uuid.uuid4()),
                    vector=embeddings[i]
                )

            self.client.batch.flush()


weaviate_url = "http://localhost:8080"
embedding_model_name = 'all-MiniLM-L6-v2'
files = get_result_files(r'C:\Project\RAG-with-Local-VectorDB-LLLM\data_ingest_pipeline\my-docs')
# Usage
weaviate_handler = WeaviateHandler(db_url=weaviate_url)
weaviate_handler.upload_schema(my_schema=weaviate_handler.create_schema())

embedding_model = SentenceTransformer(embedding_model_name, device='cpu')
data_uploader = WeaviateDataUploader(client=weaviate_handler.client)
data_uploader.add_data_to_weaviate(files=files, chunk_under_n_chars=250, chunk_new_after_n_chars=500)

print(weaviate_handler.count_documents()['data']['Aggregate']['Doc'])
