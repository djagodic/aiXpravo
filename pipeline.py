#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import unicodedata
import re

def safe_filename(text):
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return text.lower()

def load_artifacts(artifacts_dir: Path):
    metadata_path = artifacts_dir / "metadata.json"
    embeddings_path = artifacts_dir / "embeddings.npy"
    index_path = artifacts_dir / "ip_opinions.index"

    for p in [metadata_path, embeddings_path, index_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing artifact: {p}")

    with metadata_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    import numpy as np

    embeddings = np.load(embeddings_path).astype("float32")
    import faiss

    index = faiss.read_index(str(index_path))

    if len(records) != embeddings.shape[0] or index.ntotal != len(records):
        raise RuntimeError("Artifact sizes do not match (metadata, embeddings, index)")

    return records, embeddings, index


def load_manifest(artifacts_dir: Path) -> Dict[str, Any]:
    manifest_path = artifacts_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing artifact: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def encode_query(query: str, dim: int, model_name: str, cache_dir: Optional[str] = None):
    from sentence_transformers import SentenceTransformer

    import numpy as np

    model = SentenceTransformer(model_name, cache_folder=cache_dir)
    if hasattr(model, "encode_query"):
        vec = model.encode_query(query, convert_to_numpy=True, normalize_embeddings=True, truncate_dim=dim)
    else:
        prompted = f"task: search result | query: {query}"
        vec = model.encode(prompted, convert_to_numpy=True, normalize_embeddings=True, truncate_dim=dim)
    vec = np.asarray(vec, dtype="float32")
    if vec.shape[0] != dim:
        raise RuntimeError(f"Query embedding dim {vec.shape[0]} does not match index dim {dim}")
    return vec


def search_top_k(records: List[Dict[str, Any]], index: Any, query: str, k: int, model_name: str, cache_dir: Optional[str]) -> List[Dict[str, Any]]:
    dim = index.d
    qvec = encode_query(query, dim, model_name, cache_dir).reshape(1, -1)
    scores, indices = index.search(qvec, min(k, len(records)))

    out = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        rec = records[int(idx)]
        out.append(
            {
                "rank": rank,
                "score": float(score),
                "title": rec.get("title"),
                "link": rec.get("link"),
            }
        )
    return out


def run_browser(link: str, output_path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "browser.py",
            "--url",
            link,
            "--output",
            str(output_path),
        ],
        check=True,
    )


def run_test(folder: Path) -> Dict[str, Any]:
    proc = subprocess.run(
        [sys.executable, "test.py", "--data-folder", str(folder)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(proc.stdout)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end pipeline: vector search -> scrape -> contradiction test")
    parser.add_argument("--query", required=True, help="User query")
    parser.add_argument("--artifacts-dir", default="artifacts", help="Directory with vector DB artifacts")
    parser.add_argument("--top-k", type=int, default=20, help="How many nearest opinions to process")
    parser.add_argument("--output-dir", default="json_datoteke", help="Folder for extracted JSON files")
    parser.add_argument("--results-json", default="pipeline_results.json", help="Path for final pipeline result JSON")
    parser.add_argument("--cache-dir", default=None, help="Optional HF cache dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts_dir = Path(args.artifacts_dir)
    records, _embeddings, index = load_artifacts(artifacts_dir)
    manifest = load_manifest(artifacts_dir)
    model_name = manifest.get("model_name", "google/embeddinggemma-300m")

    top_results = search_top_k(records, index, args.query, args.top_k, model_name, args.cache_dir)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extracted_files = []
    for item in top_results:
        if not item.get("link"):
            continue
        safe_name = "".join(c.lower() if c.isalnum() else "_" for c in (item.get("title") or "mnenje"))[:100].strip("_")
        out_file = output_dir / f"{item['rank']:02d}_{safe_name or 'mnenje'}.json"
        run_browser(item["link"], out_file)
        extracted_files.append(str(out_file))

    contradiction = run_test(output_dir)

    payload = {
        "query": args.query,
        "top_k": args.top_k,
        "top_results": top_results,
        "extracted_files": extracted_files,
        "contradiction_check": contradiction,
    }

    with Path(args.results_json).open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
