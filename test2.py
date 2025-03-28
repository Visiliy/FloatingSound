import re
import pandas as pd
import numpy as np
import faiss
from pymorphy2 import MorphAnalyzer
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException
from pathlib import Path
import pickle

# --- Константы ---
MODEL_NAME = 'distiluse-base-multilingual-cased-v2'  # Новая модель
DATA_FILE = 'songs2.txt'
INDEX_FILE = 'faiss_index.bin'
EMBEDDINGS_FILE = 'embeddings.pkl'
METADATA_FILE = 'metadata.pkl'

# --- Предобработка текста ---
morph = MorphAnalyzer()

def preprocess(text):
    if pd.isna(text):
        return ''
    text = str(text).lower()
    # Убираем только несущественные символы, сохраняем знаки препинания
    text = re.sub(r'[^а-яё\s.,!?]', '', text)
    words = [morph.parse(word)[0].normal_form for word in text.split()]
    return ' '.join(words)

# --- Загрузка или создание базы ---
def load_or_create_data():
    if Path(INDEX_FILE).exists() and Path(EMBEDDINGS_FILE).exists() and Path(METADATA_FILE).exists():
        print("Загружаем сохранённые данные...")
        index = faiss.read_index(INDEX_FILE)
        with open(EMBEDDINGS_FILE, 'rb') as f:
            embeddings = pickle.load(f)
        with open(METADATA_FILE, 'rb') as f:
            metadata = pickle.load(f)
        return index, embeddings, metadata['titles'], metadata['texts']
    
    print("Создаём новую базу...")
    df = pd.read_csv(DATA_FILE, sep='|', header=None, names=['title', 'text'])
    df = df.drop_duplicates()  # Удаляем дубликаты
    texts = df['text'].tolist()
    titles = df['title'].tolist()
    processed_texts = [preprocess(text) for text in texts]
    
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(processed_texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')
    
    # Нормализуем эмбеддинги
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
    
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    
    # Сохраняем данные
    faiss.write_index(index, INDEX_FILE)
    with open(EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump(embeddings, f)
    with open(METADATA_FILE, 'wb') as f:
        pickle.dump({'titles': titles, 'texts': texts}, f)
    
    return index, embeddings, titles, texts

# --- Инициализация ---
index, embeddings, titles, texts = load_or_create_data()
model = SentenceTransformer(MODEL_NAME)

# --- Поиск ---
def search(query: str, top_k: int = 5) -> list[dict]:
    try:
        processed_query = preprocess(query)
        query_embedding = model.encode([processed_query])[0].astype('float32')
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        distances, indices = index.search(query_embedding.reshape(1, -1), top_k)
        
        results = []
        for i in range(top_k):
            results.append({
                'title': titles[indices[0][i]],
                'text': texts[indices[0][i]],
                'similarity': float(distances[0][i])
            })
        return results
    except Exception as e:
        print(f"Ошибка при поиске: {e}")
        return []

# --- Пример использования ---
query = input("Введите запрос: ")
results = search(query, 1)
for res in results:
    print(f"\n🎵 {res['title']} (сходство: {res['similarity']:.2f})")
    print(f"📝 {res['text'][:200]}...\n")
