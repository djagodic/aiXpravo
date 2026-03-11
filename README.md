# aiXpravo
Ai x Pravo hekathon on FMF

Workflow:
* crawler (ip-rs)
* embedding generator (prednost je: hitrejše iskanje, šparanje na tokenih, semantična podobnost, manj halucinacij)
* vector database
* cosine similarity search (PCA graf)
* top 10 mnenj (z Broser Usom bomo iz 10 najbližjih extractal ključne podatke)
* LLM povzetek (Broswer Use) <=> Tax-Fin-Lex (keywords, info laws …)
* kontradikcije (označbe)
* povzetek
* rešitev

## `embedding.py` dependencies

`embedding.py` imports and requires:

- Python standard library: `os`, `json`
- Third-party libraries: `numpy`, `faiss`, `torch`, `transformers`

### 1) Local computer (CPU)

Recommended install in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy faiss-cpu torch transformers sentencepiece safetensors huggingface_hub
```

Notes:

- `faiss-cpu` provides `import faiss` on local CPU installs.
- `sentencepiece` can be required by some Hugging Face tokenizers.
- `safetensors` and `huggingface_hub` are commonly needed by modern HF models.

### 2) Arnes gruča (CPU node)

On a typical HPC node, use user-space venv and CPU wheels:

```bash
python3 -m venv ~/venvs/aixpravo
source ~/venvs/aixpravo/bin/activate
pip install --upgrade pip
pip install numpy faiss-cpu torch transformers sentencepiece safetensors huggingface_hub
```

If your Arnes environment has no outbound internet on compute nodes, first download/install
the same wheels on a login node (or mirror) and then run `embedding.py`.

## End-to-end pipeline (query -> top 20 -> browser extraction -> contradiction check)

New script `pipeline.py` orchestrates the full flow:

1. Query the local vector DB (`artifacts_fast` by default) and select top 20 closest opinions.
2. For each result call `browser.py` with the exact opinion link.
3. Save extracted opinion JSON files to an output folder (default `json_datoteke/`).
4. Run `test.py` on that folder and produce final contradiction analysis.

### Run

```bash
export LABS_API_KEY="..."
python pipeline.py --query "vaše vprašanje" --top-k 20
```

Main outputs:
- folder with extracted JSON files (default `json_datoteke/`),
- aggregate result file (default `pipeline_results.json`).

### Notes

- `browser.py` now supports CLI usage per-link:

```bash
python browser.py --url "https://www.ip-rs.si/..." --output "json_datoteke/01_mnenje.json"
```

- `test.py` now supports selecting the source folder:

```bash
python test.py --data-folder json_datoteke
```
