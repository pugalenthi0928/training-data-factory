# Training Data Robo

An **enterprise-style training data factory** for LLMs.

- Ingest PDFs, text files, and folders of documents.
- Chunk them intelligently into paragraph-sized pieces.
- Generate synthetic training data:
  - Summaries
  - Question–Answer pairs
  - Key points
  - Titles
- Inspect, filter, and download datasets via a Streamlit dashboard.
- Export JSONL datasets you can use for:
  - Fine-tuning
  - RAG-style question answering
  - Internal analytics

---

## 1. Installation

### 1.1. Clone and create environment

```bash
git clone <your-repo-url>.git
cd training-data-factory

conda create -n myenv python=3.9 -y
conda activate myenv

pip install -e .
