from flask import *
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import numpy as np
import json
from deepgram import (
    DeepgramClient,
    PrerecordedOptions,
    FileSource,
)
from random import randrange
from flask_cors import CORS
import os
from datetime import datetime
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


app = Flask(__name__)
app.config["SECRET_KEY"] = (
    "5457fae2a71f9331bf4bf3dd6813f90abeb33839f4608755ce301b9321c671791673817685w47uer6919hdhifj"
)
CORS(app)

MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
DATA_FILE = 'songs.txt'
INDEX_FILE = 'faiss_index.bin'
EMBEDDINGS_FILE = 'embeddings.pkl'
METADATA_FILE = 'metadata.pkl'

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
    
    faiss.write_index(index, INDEX_FILE)
    with open(EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump(embeddings, f)
    with open(METADATA_FILE, 'wb') as f:
        pickle.dump({'titles': titles, 'texts': texts}, f)
    
    return index, embeddings, titles, texts

index, embeddings, titles, texts = load_or_create_data()
model = SentenceTransformer(MODEL_NAME)

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
                'similarity': float(distances[0][i])
            })
        return results
    except Exception as e:
        return False


@app.route("/get_music", methods=["POST"])
def get_music2():
    try:
        print("Ok0")
        file = request.files['audio']
  
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'recording_{timestamp}.wav'
        filepath = os.path.join("media_files", filename)

        file.save(filepath)

        deepgram = DeepgramClient("8f6543bbb44982acf9560c76d000cb55c8b8f4de")
        print("OK_test")

        with open(f"media_files/{filename}", "rb") as file:
            buffer_data = file.read()

        payload: FileSource = {
            "buffer": buffer_data,
        }

        options = PrerecordedOptions(
            model="nova-2",
            smart_format=True,
            language="ru",
        )
        response = deepgram.listen.rest.v("1").transcribe_file(payload, options)
        pattern = json.loads(response.to_json(indent=4))["results"]["channels"][0][
            "alternatives"
        ][0]["transcript"]
        print(pattern)
        print("Ok1")
        query = pattern
        results = search(query, 1)
        for res in results:
            print(f"\n🎵 {res['title']} (сходство: {res['similarity']:.2f})")
            return jsonify([res['title'], ""], 200)

    except:
        return jsonify("No", 200)
        
def main():
    app.run(port=8090)


if __name__ == "__main__":
    main()