from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    model_path: str = "/work/MLShare/rek21muv/mistral/models/mistral-7b"
    finetuned_path: str = "/work/MLShare/rek21muv/mistral/models/mistral-7b-finqa/checkpoint-1000"
    data_path: str = "/work/MLShare/rek21muv/mistral/data/finqa/dev.json"
    max_seq_length: int = 512
    max_new_tokens: int = 32
    temperature: float = 0.1
    host: str = "0.0.0.0"
    port: int = 8000
    learning_rate: float = 2e-4
    mlflow_uri: str = "sqlite:////work/MLShare/rek21muv/mistral/mlflow.db"

    class Config:
        env_file = ".env"

settings = Settings()