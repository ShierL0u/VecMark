VecMark
=======

The repo provides:

- Watermark embedding for datasets
- Dataset- and RAG-level detection/evaluation
- Attacks and visualization utilities

All code is organized to use relative paths only

Directory Structure
-------------------

```
VecMark/
  Attack/
    RAG_attack/           # RAG attacks Experiments.
    structure_attack/     # Subset and structure attacks Experiments.
  KGW/                    # KGW
  data/
    attacked_eva/         # Attack evaluation outputs (JSON/figures)
    mapping/              # Rewrite/QA mappings generated during embedding/QA
    rag_result/           # Saved RAG QA results for alignment evaluation
    transformed_dataset/  # Intermediate transformed datasets
    watermarked_data/     # Final watermarked datasets
  main/
    emb.py                # Watermark embedding entry
    evalution.py          # RAG/Dataset evaluation entry (PPL/RS/CDPA/CIRA)
    det_dataset.py        # Dataset-level watermark detection (text-aware)
    det_RAG.py            # RAG-level watermark detection
    gamma_search.py       # Utilities for gamma search
    utils1.py             # Shared helpers (hashing, model loaders, prompts)
  vector_database/        # Chroma persistence directories (auto-created)
  __init__.py
```

Datasets
--------

Place datasets next to the project as:

```
../dataset/
  nfcorpus_corpus.csv
  winemag_sub_dataset_5k.csv
  winemag_sub_dataset_50k.csv
  winemag-data-130k-v2.csv
  FCT.csv
  FCT_100k.csv
  watermark_dataset/
    VectorMark_nfcorpus_g37.csv
    VectorMark_nfcorpus_g37_rewrite_mapping.json
    VectorMark_winemag50k_g587.csv
    VectorMark_winemag50k_g587_rewrite_mapping.json
    (and other method outputs like RAGWM_*.csv, WARD_*.csv)
  attacked_dataset/       # Auto-created by attack scripts
```

All code reads and writes via relative paths from the project root, so the sibling `../dataset` folder is expected by default. You may change the base directory by editing script arguments or small constants where noted below.

Models
------

The project loads local models by relative paths, e.g.:

- `../model/Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8` for HF transformers
- `ollama` backends via `langchain_ollama` (e.g., `qwen:14b`, `deepseek-r1:latest`, `bge-m3:latest`)

Adjust these names in the corresponding script arguments or small constants if your local model names differ.

Quick Start
-----------

0. choose a suitable gamma : gamma_search.py
   
   and ensure that an appropriate modulus is selected to prevent collisions from occurring with this key.
1) Embed watermarks on a dataset

```
python main/emb.py
```

Key outputs:

- Watermarked CSV in `data/watermarked_data/`

- Rewrite mapping JSON in `data/mapping/`
2) Dataset-level detection

```
python main/det_dataset.py
```

This runs a text-aware detector that only evaluates sentences likely containing watermark edits.

3) RAG evaluation (Perplexity/Rationality Score) and Alignment (CDPA/CIRA)

```
python main/evalution.py
```

Within the script, two helpers are provided:

- `test_onRAG_PPL_RS(...)` to compute PPL (optional) and RS for different methods
- `test_onRAG_CDPA_CIRA(...)` to compute alignment metrics based on generated RAG results

The code automatically persists intermediate RAG databases under `vector_database/` and saves results to `data/rag_result/`.

4) attacks and figures

Dataset-level and RAG-level attacks are under `Attack/`:

- `Attack/structure_attack/*` for dataset perturbations and plotting
- `Attack/RAG_attack/*` for RAG pipeline perturbations

Run an attack entry script to generate attacked datasets and evaluation curves; figures and JSON summaries are written under `data/attacked_eva/`.
