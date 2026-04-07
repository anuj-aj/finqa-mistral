# FinQA Mistral — Financial QA with Fine-tuned Mistral 7B

A production-grade financial question answering system built by fine-tuning Mistral 7B using LoRA (QLoRA) on the FinQA dataset. Served via FastAPI and containerized with Docker.

## Setup

git clone <repo>
cd mistral
conda create -n finqa-mistral python=3.10 -y
conda activate finqa-mistral
pip install -r requirements.txt
pip install -e .

## Configuration

All configuration via `.env`:
```env
MODEL_PATH=/models/mistral-7b
FINETUNED_PATH=/models/mistral-7b-finqa-v2
DATA_PATH=/data/finqa/dev.json
MAX_SEQ_LENGTH=512
MAX_NEW_TOKENS=16
TEMPERATURE=0.1
PORT=8000
```

## Training
```bash
sbatch scripts/slurm/train.sh
```

## Evaluation
```bash
sbatch scripts/slurm/evaluate.sh
```

## Inference

### Local
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Docker
```bash
docker-compose up --build
```

### API
```bash
curl -X POST http://localhost:8000/api/v1/predict \\
  -H "Content-Type: application/json" \\
  -d '{"context": "Revenue was 12.3B in 2018 and 14.1B in 2019.", "question": "What was revenue in 2019?"}'
```

## Interactive API Docs

| Interface | URL |
|-----------|-----|
| Swagger UI | http://<host>:8000/docs |
| ReDoc | http://<host>:8000/redoc |

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | Health check |
| POST | /api/v1/predict | Financial QA inference |

## Model Details

| Parameter | Value |
|-----------|-------|
| Base Model | Mistral 7B v0.1 |
| Fine-tuning | QLoRA (4-bit) |
| LoRA Rank | 16 |
| LoRA Alpha | 32 |
| Target Modules | q_proj, v_proj |
| Dataset | FinQA (6,251 train / 883 dev) |
| Epochs | 5 (3 + 2 resumed) |
| Batch Size | 16 (effective) |



## for MLflow
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
