# test_recommendations.py
import pandas as pd
import joblib

# --- Configuração ---
MODEL_PATH = 'models/svd_model.pkl'
DATA_PATH = 'data/ratings.csv' # Usado para saber quais filmes o usuário NÃO viu
USER_ID_TO_TEST = 1 # Escolha um ID de usuário para testar
TOP_N = 10 # Quantas recomendações queremos ver

# 1. Carregar o modelo treinado
print(f"Carregando modelo de '{MODEL_PATH}'...")
model = joblib.load(MODEL_PATH)
print("Modelo carregado.")

# 2. Carregar os dados para encontrar filmes não vistos pelo usuário
df = pd.read_csv(DATA_PATH)

# Pega a lista de todos os IDs de filmes
all_movie_ids = df['movieId'].unique()

# Pega a lista de filmes que o usuário JÁ avaliou
movies_rated_by_user = df[df['userId'] == USER_ID_TO_TEST]['movieId'].unique()

# Encontra os filmes que o usuário AINDA NÃO avaliou
movies_to_predict = [movie_id for movie_id in all_movie_ids if movie_id not in movies_rated_by_user]
print(f"Encontrados {len(movies_to_predict)} filmes para gerar previsões para o usuário {USER_ID_TO_TEST}.")

# 3. Gerar previsões para todos os filmes não vistos
print("Gerando previsões...")
predictions = []
for movie_id in movies_to_predict:
    # A mágica acontece aqui: model.predict()
    predicted_rating = model.predict(uid=USER_ID_TO_TEST, iid=movie_id).est
    predictions.append((movie_id, predicted_rating))

# 4. Ordenar as previsões e mostrar as TOP N melhores
predictions.sort(key=lambda x: x[1], reverse=True)
top_n_recommendations = predictions[:TOP_N]

print(f"\n--- TOP {TOP_N} Recomendações para o Usuário {USER_ID_TO_TEST} ---")
for movie_id, estimated_rating in top_n_recommendations:
    print(f"Filme ID: {movie_id}, Nota Prevista: {estimated_rating:.2f}")