from flask import *
import json
import json
import os
from fuzzywuzzy import fuzz
import re
import sqlite3
from deepgram import (
    DeepgramClient,
    PrerecordedOptions,
    FileSource,
)
from random import randrange
from flask_cors import CORS
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash


app = Flask(__name__)
app.config["SECRET_KEY"] = (
    "5457fae2a71f9331bf4bf3dd6813f90abeb33839f4608755ce301b9321c671791673817685w47uer6919hdhifj"
)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"
db = SQLAlchemy(app)
CORS(app)


class Users(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    nickname = db.Column(db.String())
    password = db.Column(db.String())
    favoritemusic = db.Column(db.Text())

    def __repr__(self):
        return f"<users {self.id}>"
    

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
    

@app.route("/login", methods=["POST"])
def login():
    reg = request.get_json()
    user = Users.query.filter_by(nickname=reg["name"]).first()
    if user:
        if check_password_hash(user.password, reg["password"]):
            return jsonify([True, True], 200)
        else:
            return jsonify([True, False], 200)
    return jsonify([False, False], 200)
    


@app.route("/registration", methods=["POST"])
def registration():
    reg = request.get_json()
    print(reg)
    if Users.query.filter_by(nickname=reg["name"]).first():
        return jsonify(False, 200)
    users = Users(
        nickname=reg["name"],
        password=generate_password_hash(reg["password"]),
        favoritemusic=json.dumps([]),
    )
    db.session.add(users)
    db.session.flush()
    db.session.commit()
    return jsonify(True, 200)


@app.route("/get_music", methods=["POST"])
def get_music2():
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
    songs = load_songs()
    found_songs = fuzzy_search(songs, pattern, threshold=60)
    res = display_results(found_songs)
    print(res)
    return jsonify(res, 200)
        
def main():
    # with app.app_context():
    #     db.create_all()
    app.run(port=8090)


if __name__ == "__main__":
    main()