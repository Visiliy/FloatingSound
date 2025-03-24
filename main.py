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


app = Flask(__name__)
app.config["SECRET_KEY"] = (
    "5457fae2a71f9331bf4bf3dd6813f90abeb33839f4608755ce301b9321c671791673817685w47uer6919hdhifj"
)
CORS(app)


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
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

        documents = []
        for q in range(1, 1293):
            with open(f"audio1/media{q}.txt", "r", encoding="utf-8") as f:
                text = f.read()
                chank_list = []
                s = ".!?:;,"
                big_chank = ""
                for i in text:
                    if i in s:
                        if i in s[:-1]:
                            big_chank += "/"
                        else:
                            big_chank += " "
                    else:
                        big_chank += i
                chank_list = big_chank.split("/")
                for k in chank_list:
                    if k == "":
                        del chank_list[chank_list.index("")]
                documents += chank_list
        print("Ok3")
        query = pattern

        with open("doc.txt", "r", encoding="utf-8") as f2:
            h_m = json.loads(f2.read())
        document_embeddings_loaded = np.load('document_embeddings.npy')
        query_embedding = model.encode(query)
        similarities = cosine_similarity([query_embedding], document_embeddings_loaded)[0]
        sorted_indices = np.argsort(similarities)[::-1]
        d = documents[sorted_indices[0]]
        os.remove(f"media_files/{filename}")
        print("OK4")
        for g in h_m:
            for a in h_m[g]:
                if a == d:
                    print(g)
                    return jsonify([g, ""], 200)
        return jsonify("Ничего не найдено", 200)
    except:
        return jsonify("No", 200)
        
def main():
    app.run(port=8090)


if __name__ == "__main__":
    main()