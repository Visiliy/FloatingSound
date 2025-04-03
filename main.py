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
from music2notes import vocall


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
        title, text, soong = i[1], i[2], i[3]
        songs.append({
            'title': title.strip(),
            'text': text.strip(),
            'processed_text': preprocess_text(text),
            'soong': soong
                    })
    connection.close()
    return songs


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def fuzzy_search(songs, query, notes, threshold=40, limit=10):
    processed_query = preprocess_text(query)
    print(processed_query)
    print(len(processed_query))
    print(notes)
    print(len(notes))
    results = []
    n, t = 1, 1
    if len(processed_query) > 20:
        n = 0.4
    elif 0 < len(processed_query) < 10:
        t = 0.4
    elif len(processed_query) == 0:
        t = 0
    if len(notes) < 15:
        n = 0.4

    print(t, n)

    for song in songs:
        
        text_ratio = fuzz.partial_ratio(processed_query, song['processed_text'])
        notes_ratio = fuzz.partial_ratio(notes, song['soong'])

        combined_score = text_ratio * t + notes_ratio * n
        if combined_score >= threshold:
            results.append({
                'song': song,
                'combined_score': combined_score
            })

    results.sort(key=lambda x: x['combined_score'], reverse=True)
    return results[0]['song']['title']


def display_results(results):
    if not results:
        return "Ничего не найдено."
    
    return results
    

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
    notes = ' '.join(vocall(filepath))
    print(notes)

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
    found_songs = fuzzy_search(songs, pattern, notes, threshold=50)
    res = display_results(found_songs)
    print(res)
    return jsonify(res, 200)
        
def main():
    # with app.app_context():
    #     db.create_all()
    app.run(port=8090)


if __name__ == "__main__":
    main()


# ('Гимн Российской Федерации', '5 -5 2 2 -7 5 -2 -2 2 -7 2 2 1 2 2 2 1 2 -7 9 -2 -2 2 -3 -4 5 -1 -2 2 -7 5 -2 -2 2 -7 12 -1 -2 -2')
# ('В лесу родилась ёлочка', '9 -2 2 -4 -5 9 1 -3 5 -10 8 -1 -2 -2 -5 9 -2 2 -4 7 -10 8 -1 -2 -2 -5 9 -2 2 -4')
# ('А знаешь, всё ещё будет', '9 -2 2 1 -1 -9 9 -2 2 -2 -2 1 1 2 3 -2 -1 -2 -2 2 -7 9 -2 2 1 -1 -9 9 -2 2 -2 -2 1 1 2 3 -2 -1 -2 -2 2 2 -2 -2')
# ('Группа крови', '-1 -2 7 -7 2 1 -1 5 -5 1 -3 3 -3 2 1 -1 -4 -3 8 -1 1 -3 2 1 -1 5 -5 1 -3 3 -3 2 1 -1 5 -2 -2 -1 1 -1 -2')
# ('К Элизе', '-1 1 -1 1 -5 3 -2 -3 -5 5 2 -7 5 2 1 -8 12 -1 1 -1 1 -5 3 -2 -3 -5 5 2 -7 8 -1 -2 2 1 2 2 -9 10 -1 -2 -9 11 -2 -2 2 -2 -1 -7 8 -1 -2')
