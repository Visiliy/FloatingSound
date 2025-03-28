import re
import pandas as pd
import numpy as np
import faiss
from pymorphy2 import MorphAnalyzer
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException
from pathlib import Path
import pickle

# --- Константы ---
MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
DATA_FILE = 'songs2.txt'
INDEX_FILE = 'faiss_index.bin'
EMBEDDINGS_FILE = 'embeddings.pkl'
METADATA_FILE = 'metadata.pkl'

# --- Предобработка текста ---
morph = MorphAnalyzer()
stop_words = {
    'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 
    'то', 'все', 'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же',
    'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне', 'было', 'вот', 'от',
    'меня', 'еще', 'нет', 'о', 'из', 'ему', 'теперь', 'когда', 'даже',
    'ну', 'вдруг', 'ли', 'если', 'уже', 'или', 'ни', 'быть', 'был', 
    'него', 'до', 'вас', 'нибудь', 'опять', 'уж', 'вам', 'ведь', 'там', 
    'потом', 'себя', 'ничего', 'ей', 'может', 'они', 'тут', 'где', 'есть',
    'надо', 'ней', 'для', 'мы', 'тебя', 'их', 'чем', 'была', 'сам', 'чтоб',
    'без', 'будто', 'чего', 'раз', 'тоже', 'себе', 'под', 'будет', 'ж', 
    'тогда', 'кто', 'этот', 'того', 'потому', 'этого', 'какой', 'совсем',
    'ним', 'здесь', 'этом', 'один', 'почти', 'мой', 'тем', 'чтобы', 'нее',
    'сейчас', 'были', 'куда', 'зачем', 'всех', 'никогда', 'можно', 'при',
    'наконец', 'два', 'об', 'другой', 'хоть', 'после', 'над', 'больше',
    'тот', 'через', 'эти', 'нас', 'про', 'всего', 'них', 'какая', 'много',
    'разве', 'три', 'эту', 'моя', 'впрочем', 'хорошо', 'свою', 'этой',
    'перед', 'иногда', 'лучше', 'чуть', 'том', 'нельзя', 'такой', 'им',
    'более', 'всегда', 'конечно', 'всю', 'между'
}

def preprocess(text):
    if pd.isna(text):
        return ''
    text = str(text).lower()
    text = re.sub(r'[^а-яё\s]', '', text)
    words = [morph.parse(word)[0].normal_form for word in text.split() if word not in stop_words]
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
    texts = df['text'].tolist()
    titles = df['title'].tolist()
    processed_texts = [preprocess(text) for text in texts]
    
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(processed_texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')
    
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
        query_embedding = model.encode([processed_query])[0].astype('float32').reshape(1, -1)
        distances, indices = index.search(query_embedding, top_k)
        
        results = []
        for i in range(top_k):
            results.append({
                'title': titles[indices[0][i]],
                'text': texts[indices[0][i]],
                'similarity': float(distances[0][i])  # Для JSON-сериализации
            })
        return results
    except Exception as e:
        return False


query = input("Введите запрос: ")
results = search(query, 1)
for res in results:
    print(f"\n🎵 {res['title']} (сходство: {res['similarity']:.2f})")
    print(f"📝 {res['text'][:200]}...\n")