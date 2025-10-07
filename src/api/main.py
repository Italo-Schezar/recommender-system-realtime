# src/api/main.py

from fastapi import FastAPI
import joblib
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = ROOT_DIR / "models" / "svd_model.pkl"
DATA_PATH = ROOT_DIR / "data" / "ratings.csv"
TOP_N = 10

app = FastAPI(title="API de Sistema de Recomendação")

model = joblib.load(MODEL_PATH)
df_ratings = pd.read_csv(DATA_PATH)
# AJUSTADO DE VOLTA PARA 'movieId'
all_movie_ids = df_ratings['movieId'].unique()

print(f"API pronta. Modelo carregado de '{MODEL_PATH}'")

@app.get("/")
def read_root():
    return {"message": "Bem-vindo à API de Sistema de Recomendação!"}

@app.get("/recommendations/{user_id}")
def get_recommendations(user_id: int, top_n: int = TOP_N):
    if user_id not in df_ratings['userId'].unique():
        return {"error": "Usuário não encontrado"}, 404

    # AJUSTADO DE VOLTA PARA 'userId' e 'movieId'
    movies_rated_by_user = df_ratings[df_ratings['userId'] == user_id]['movieId'].unique()
    movies_to_predict = [movie_id for movie_id in all_movie_ids if movie_id not in movies_rated_by_user]
    
    predictions = []
    for movie_id in movies_to_predict:
        predicted_rating = model.predict(uid=user_id, iid=movie_id).est
        # AJUSTADO DE VOLTA PARA 'movieId'
        predictions.append({"movie_id": movie_id, "estimated_rating": predicted_rating})
        
    predictions.sort(key=lambda x: x["estimated_rating"], reverse=True)
    top_n_recommendations = predictions[:top_n]

    return {
        "user_id": user_id,
        "recommendations": top_n_recommendations
    }