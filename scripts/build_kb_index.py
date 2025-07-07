import argparse
import json
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def build_index(input_path: str, out_dir: str, model_name: str):
    # Load facts/rules
    with open(input_path, 'r', encoding='utf-8') as f:
        docs = [json.loads(line.strip())["text"] for line in f if line.strip()]

    # Create output directory if not exists
    os.makedirs(out_dir, exist_ok=True)

    # Load embedding model
    model = SentenceTransformer(model_name)
    embeddings = model.encode(docs, show_progress_bar=True)

    # Save jsonl (for later retrieval)
    with open(os.path.join(out_dir, "index.jsonl"), "w", encoding='utf-8') as f_out:
        for doc in docs:
            json.dump({"text": doc}, f_out)
            f_out.write("\n")

    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings, dtype="float32"))

    # Save FAISS index
    faiss.write_index(index, os.path.join(out_dir, "index.faiss"))
    print(f"Saved FAISS index with {len(docs)} documents to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to facts_rules_full.jsonl")
    parser.add_argument("--out_dir", required=True, help="Output directory to save index files")
    parser.add_argument("--model", default="BAAI/bge-small-en", help="Embedding model to use")

    args = parser.parse_args()
    build_index(args.input, args.out_dir, args.model)
