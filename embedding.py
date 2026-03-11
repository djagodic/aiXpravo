import os
import json
import numpy as np
import faiss
import torch
from transformers import AutoTokenizer, AutoModel

DATA_DIR = "database"

model_name = "google/embeddinggemma-300m"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


def embed(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)

    embedding = outputs.last_hidden_state.mean(dim=1)

    return embedding[0].numpy()

def load_category(category_path):

    titles = []
    links = []

    for file in os.listdir(category_path):

        if file.endswith(".json"):

            path = os.path.join(category_path, file)

            with open(path, "r", encoding="utf-8") as f:

                data = json.load(f)

                titles.append(data["title"])
                links.append(data["link"])

    return titles, links

def build_index(category_name, titles):

    vectors = []

    for t in titles:
        vectors.append(embed(t))

    vectors = np.array(vectors).astype("float32")

    # normalizacija za cosine similarity
    faiss.normalize_L2(vectors)

    dimension = vectors.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(vectors)

    print(f"{category_name} -> {index.ntotal} dokumentov")

    faiss.write_index(index, f"{category_name}.index")

    return index

def build_all_databases():

    metadata = {}

    for category in os.listdir(DATA_DIR):

        category_path = os.path.join(DATA_DIR, category)

        if os.path.isdir(category_path):

            titles, links = load_category(category_path)

            build_index(category, titles)

            metadata[category] = [
                {"title": t, "link": l}
                for t, l in zip(titles, links)
            ]

    with open("metadata.json", "w", encoding="utf-8") as f:

        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("Baze zgrajene!")

def search(query, category, k=5):

    index = faiss.read_index(f"{category}.index")

    with open("metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    qvec = embed(query).astype("float32")

    faiss.normalize_L2(qvec.reshape(1, -1))

    D, I = index.search(np.array([qvec]), k)

    print("\nRezultati:\n")

    for i in I[0]:

        item = metadata[category][i]

        print(item["title"])
        print(item["link"])
        print()

if __name__ == "__main__":

    # zgradi vse baze
    build_all_databases()

    # primer iskanja
    #query = "Ali lahko delodajalec uporablja biometrične podatke?"

    #search(query, "kategorija1", 5)