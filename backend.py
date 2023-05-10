from flask import Flask, request
from gensim.models import Word2Vec

app = Flask(__name__)

word2vec_model = Word2Vec.load('model.bin')

@app.after_request
def after_request(response):
  response.headers.add('Access-Control-Allow-Origin', '*')
  response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
  response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
  return response


@app.route("/process-input", methods=["POST"])
def process_input():

    input_text = request.json["inputText"]

    ingredients = input_text.split(',')
    similar_ingredients = []

    for ingredient in ingredients:
        try:
            similar_ingredients.extend(word2vec_model.wv.most_similar(ingredient.lstrip().rstrip(), topn=1))
        except:
            continue

    # extract the ingredient names from the Word2Vec output
    similar_ingredient_names = [x[0] for x in similar_ingredients]

    if len(similar_ingredient_names) == 0:
        return "Sorry we could not found any similar ingredients to the ones you gave us."
    else:
        return ', '.join(similar_ingredient_names)


if __name__ == "__main__":
    app.run()
