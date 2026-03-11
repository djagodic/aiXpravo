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
