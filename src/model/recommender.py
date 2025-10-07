import pandas as pd
import joblib # Usado para serializar/salvar o modelo
from surprise import Dataset, Reader
from surprise.prediction_algorithms import SVD 
import os
import logging

logging.basicConfig(level=logging.INFO)

class RecommenderModel:
    def __init__(self, model_path=None):
        """
        Inicializa o modelo e define o caminho do arquivo do modelo.
        """
        self.model = None
        if model_path is None:
            self.model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/svd_model.pkl"))
        else:
            self.model_path = model_path
    
    def train(self, data_path=None):
        """
        Carrega os dados e treina o modelo SVD.
        """
        if data_path is None:
            data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/ratings.csv"))
        logging.info("Iniciando treinamento do modelo...")
        try:
            df = pd.read_csv(data_path)
        except FileNotFoundError:
            logging.error(f"Arquivo de dados não encontrado em {data_path}")
            raise
        except Exception as e:
            logging.error(f"Erro ao ler o arquivo CSV: {e}")
            raise
        # Validação das colunas
        expected_cols = {'userId', 'movieId', 'rating'}
        if not expected_cols.issubset(df.columns):
            logging.error(f"CSV deve conter as colunas: {expected_cols}")
            raise ValueError(f"CSV deve conter as colunas: {expected_cols}")
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)
        trainset = data.build_full_trainset()
        self.model = SVD()
        self.model.fit(trainset)
        logging.info("Treinamento concluído!")
    
    def save_model(self):
        """
        Salva o modelo treinado em disco.
        """
        if self.model:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump(self.model, self.model_path)
            logging.info(f"Modelo salvo com sucesso em {self.model_path}")
        else:
            logging.error("Nenhum modelo treinado para salvar.")
            raise ValueError("Nenhum modelo treinado para salvar.")
    
    def load_model(self):
        """
        Carrega o modelo treinado do disco (usado pela API).
        """
        try:
            self.model = joblib.load(self.model_path)
            logging.info("Modelo carregado com sucesso.")
        except FileNotFoundError:
            logging.warning(f"Arquivo de modelo não encontrado em {self.model_path}. Treine o modelo primeiro.")
            raise
        except Exception as e:
            logging.error(f"Erro ao carregar o modelo: {e}")
            raise
    
    def predict_user(self, user_id, item_id):
        """
        Gera uma previsão de rating para um usuário e item específicos.
        """
        if self.model:
            try:
                prediction = self.model.predict(str(user_id), str(item_id))
                return prediction.est
            except Exception as e:
                logging.error(f"Erro na previsão: {e}")
                return None
        logging.warning("Modelo não carregado ou treinado.")
        return None