from deepface import DeepFace
from elasticsearch import Elasticsearch
import uuid

client = Elasticsearch(hosts=["http://localhost:9200"])


def indexFace(embedding: list[float], name: str, i):
    client.index(
        index="faces",
        id=uuid.uuid4().hex,
        document={"face_vector": embedding, "name": name},
    )


def queryFace(embedding: list[float]):
    response = client.search(
        index="faces",
        knn={
            "field": "face_vector",
            "query_vector": embedding,
            "k": 10,
            "num_candidates": 10,
        },
    )


def createIndex():
    client.indices.create(
        index="faces",
        mappings={
            "properties": {
                "face_vector": {
                    "type": "dense_vector",
                    "dims": 4096,
                    "index": "true",
                    "similarity": "cosine",
                },
                "name": {"type": "text"},
            }
        },
    )


backend = "yolov8"
model = "Facenet"
metric = "cosine"

embbeding = DeepFace.represent(
    img_path="/home/lav/Pictures/fotos/eu.jpg",
    model_name=model,
    detector_backend=backend,
)
createIndex()
print(queryFace(embbeding))
