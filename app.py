import flask
from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('stsb-roberta-large')
app = Flask(__name__)


@app.route('/form')
def form():
    return render_template('form.html')


@app.route('/predict', methods=["GET", "POST"])
def isDuplicate():
    sentences = []
    sentences.append(request.form.get("fname"))
    sentences.append(request.form.get("lname"))
    print(sentences)
    embeddings = model.encode(sentences, convert_to_tensor=True)
    similarity = []
    for i in range(len(sentences)):
        row = []
        for j in range(len(sentences)):
            print(j)
            row.append(util.pytorch_cos_sim(embeddings[i], embeddings[j]).item())
        similarity.append(row)
    
    similarity_percentage = round(similarity[0][1] * 100, 2)
    return "Your sentences are {}% similar.".format(similarity_percentage)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
