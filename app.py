from flask import Flask, jsonify, render_template, request
from generate_tokens import get_embedding
from append_pool import load_pool, tokenise_append_to_pool
from k_nearest import cosine_similarity

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit_text():
    # Check if the request contains JSON data
    if request.is_json:
        # Get the JSON data from the request
        data = request.get_json()
        
        description = data['description']
        
        tokenise_append_to_pool(text=description)

        # For this example, we'll just return it in the response
        return jsonify({"message": "Data received"}), 200
    else:
        return jsonify({"error": "Request must be JSON"}), 400

@app.route('/api/search', methods=['GET'])
def search():
    # Check if the request contains JSON data
    if request.is_json:
        # Get the JSON data from the request
        data = request.get_json()
        
        n_items = data['n_items']
        to_search = data['to_search']

        data = cosine_similarity(to_search, n_items)
        
        # For this example, we'll just return it in the response
        return jsonify({"message": "Data received", "data": data}), 200
    else:
        return jsonify({"error": "Request must be JSON"}), 400

if __name__ == '__main__':
    app.run(debug=True)