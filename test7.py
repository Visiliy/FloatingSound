from fuzzywuzzy import fuzz
import re
import sqlite3


def load_songs():
    songs = []
    connection = sqlite3.connect('music.db')
    cursor = connection.cursor()
    cursor.execute('SELECT * FROM Music')
    music = cursor.fetchall()
    for i in music:
        title, text = i[1], i[2]
        songs.append({
            'title': title.strip(),
            'text': text.strip(),
            'processed_text': preprocess_text(text)
                    })
    connection.close()
    return songs


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def fuzzy_search(songs, query, threshold=70, limit=10):
    processed_query = preprocess_text(query)
    results = []

    for song in songs:
        
        text_ratio = fuzz.partial_ratio(processed_query, song['processed_text'])
        
        combined_score = text_ratio
        
        if combined_score >= threshold:
            results.append({
                'song': song,
                'text_score': text_ratio,
                'combined_score': combined_score
            })

    results.sort(key=lambda x: x['combined_score'], reverse=True)
    return results[:limit]


def display_results(results):
    if not results:
        return "Ничего не найдено."
    
    return results[-1]["song"]["title"]


if __name__ == "__main__":
    songs = load_songs()
    search_query = input("\nВведите фразу для поиска (или 'q' для выхода): ")
    found_songs = fuzzy_search(songs, search_query, threshold=60)
    print(display_results(found_songs))