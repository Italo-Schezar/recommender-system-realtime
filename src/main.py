# src/api/main.py

from fastapi import FastAPI
import joblib
import pandas as pd
from pathlib import Path

# --- Configurações de Caminho Robustas ---
# Esta linha descobre o caminho absoluto para a pasta raiz do projeto
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# Agora, construímos os caminhos para os arquivos a partir da raiz
MODEL_PATH = ROOT_DIR / "models" / "svd_model.pkl"
DATA_PATH = ROOT_DIR / "data" / "ratings.csv"
TOP_N = 10

# --- Inicialização da Aplicação ---
app = FastAPI(
    title="API de Sistema de Recomendação",
    description="Uma API para servir recomendações de filmes em tempo real usando um modelo SVD.",
    version="1.0.0"
)

# --- Carregamento de Recursos (Modelo e Dados) ---
# Carrega o modelo treinado UMA VEZ quando a API inicia
model = joblib.load(MODEL_PATH)
# Carrega o dataframe de ratings para encontrar filmes não vistos
df_ratings = pd.read_csv(DATA_PATH)
# Usa 'movieId' como no arquivo CSV do MovieLens
all_movie_ids = df_ratings['movieId'].unique()

print(f"API pronta. Modelo carregado de '{MODEL_PATH}'")

# --- Endpoints da API ---
@app.get("/")
def read_root():
    return {"message": "Bem-vindo à API de Sistema de Recomendação!"}


@app.get("/recommendations/{user_id}")
def get_recommendations(user_id: int):
    """
    Gera recomendações personalizadas para um determinado ID de usuário.
    """
    # Usa 'userId' e 'movieId' como no arquivo CSV do MovieLens
    movies_rated_by_user = df_ratings[df_ratings['userId'] == user_id]['movieId'].unique()
    movies_to_predict = [movie_id for movie_id in all_movie_ids if movie_id not in movies_rated_by_user]
    
    predictions = []
    for movie_id in movies_to_predict:
        # O modelo 'surprise' internamente espera 'uid' e 'iid'
        predicted_rating = model.predict(uid=user_id, iid=movie_id).est
        # O retorno do JSON usará 'movieId' para consistência
        predictions.append({"movie_id": movie_id, "estimated_rating": predicted_rating})
        
    predictions.sort(key=lambda x: x["estimated_rating"], reverse=True)
    top_n_recommendations = predictions[:TOP_N]

    return {
        "user_id": user_id,
        "recommendations": top_n_recommendations
    }