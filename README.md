# API de Sistema de Recomendação em Tempo Real

Este projeto, `recommender-system-realtime`, é um microsserviço de alto desempenho desenvolvido em Python que serve recomendações de itens em tempo real. Ele utiliza um modelo de **Filtragem Colaborativa** com **SVD (Singular Value Decomposition)** e foi construído com foco em boas práticas de Engenharia de Software (POO) e MLOps (Modelo em Produção).

## ✨ Principais Funcionalidades

- **API de Alta Performance:** Construída com FastAPI para fornecer respostas de baixa latência, ideal para aplicações em tempo real.
- **Modelo SVD:** Utiliza o robusto algoritmo SVD da biblioteca `scikit-surprise` para gerar previsões de rating e recomendações.
- **Arquitetura Modular:** O código é organizado de forma limpa, separando a lógica do modelo (`src/model`) da lógica da API (`src/api`), facilitando a manutenção e escalabilidade.
- **Documentação Interativa:** A API gera automaticamente uma documentação interativa (Swagger UI) em `/docs`, permitindo testar os endpoints diretamente do navegador.

## 🛠️ Arquitetura e Tecnologias Utilizadas

| Componente                  | Ferramenta Principal                                |
| --------------------------- | --------------------------------------------------- |
| **Linguagem** | Python 3.11.9                                       |
| **Framework da API** | FastAPI                                             |
| **Servidor ASGI** | Uvicorn                                             |
| **Machine Learning** | Scikit-surprise                                     |
| **Manipulação de Dados** | Pandas                                              |
| **Serialização do Modelo** | Joblib                                              |
| **Gerenciamento de Ambiente** | Pyenv & Venv                                        |
| **Versionamento** | Git & GitHub                                        |

## 🚀 Configuração e Instalação

Siga os passos abaixo para configurar e executar o projeto localmente.

**1. Clonar o Repositório**
```bash
git clone [https://github.com/SEU_USUARIO/recommender-system-realtime.git](https://github.com/SEU_USUARIO/recommender-system-realtime.git)
cd recommender-system-realtime
```
*(Lembre-se de substituir `SEU_USUARIO` pelo seu nome de usuário do GitHub)*

**2. Configurar o Ambiente Python**
É recomendado usar `pyenv` para garantir a versão correta do Python.
```bash
# Instala a versão do Python, se necessário
pyenv install 3.11.9

# Cria e ativa o ambiente virtual
python3.11 -m venv venv
source venv/bin/activate
```

**3. Instalar as Dependências**
Todas as bibliotecas necessárias estão listadas no arquivo `requirements.txt`.
```bash
pip install -r requirements.txt
```

## ▶️ Como Usar

**1. Treinar o Modelo de Machine Learning**
Antes de iniciar a API, você precisa treinar o modelo com os dados disponíveis. Este script irá gerar o arquivo `models/svd_model.pkl`.
```bash
python -m tests.test_recommender
```
*(Este passo é necessário apenas uma vez ou quando os dados de `data/ratings.csv` forem atualizados).*

**2. Iniciar a API**
Com o modelo treinado, inicie o servidor FastAPI com o Uvicorn.
```bash
uvicorn src.api.main:app --reload
```
O servidor estará rodando em `http://127.0.0.1:8000`.

## 📡 Endpoints da API

A API fornece uma documentação interativa completa. Para testar, acesse **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)** no seu navegador.

### `GET /`
- **Descrição:** Endpoint raiz que retorna uma mensagem de boas-vindas.
- **Resposta de Exemplo:**
  ```json
  {
    "message": "Bem-vindo à API de Sistema de Recomendação!"
  }
  ```

### `GET /recommendations/{user_id}`
- **Descrição:** Retorna uma lista com as 10 melhores recomendações de itens para um ID de usuário específico.
- **Parâmetro:**
  - `user_id` (integer): O ID do usuário para o qual gerar as recomendações.
- **Resposta de Exemplo (para `user_id=1`):**
  ```json
  {
    "user_id": 1,
    "recommendations": [
      {
        "movie_id": 318,
        "estimated_rating": 5
      },
      {
        "movie_id": 858,
        "estimated_rating": 5
      }
    ]
  }
  ```
