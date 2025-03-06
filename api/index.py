import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load pre-trained T5 model and tokenizer
model_name = os.environ["MODEL_NAME"]
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

@app.route('/gen', methods=['POST'])
def generate_text():
    input_text = request.json['input_text']
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output = model.generate(input_ids, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2)
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return jsonify({'output_text': output_text})

if __name__ == '__main__':
    app.run(debug=True)





# from duckduckgo_search import DDGS
# import requests
# from flask import Flask, request, jsonify
# from flask_cors import CORS


 
# app = Flask(__name__)

# CORS(app, origins=["https://search-beta-six.vercel.app"])

# @app.route("/search", methods=['GET'])

# def search():
#         # Initialize DDGS
#         query = request.args.get('q')
#         with DDGS() as ddgs:
#         # Perform a text search
#             results = ddgs.text(query, region="wt-wt", safesearch="moderate", timelimit="y")

#             Request_List = []
#     # Print results
#         for result in results:
#             # print(f"Title: {result['title']}")
#             # print(f"URL: {result['href']}")
#             # print(f"Snippet: {result['body']}")
#             # print("-" * 1)
#             Request_List.append({"title": result['title'], "url": result['href'], "snippet": result['body']})

#         return jsonify(Request_List);



        
# if __name__ == '__main__':
#     app.run(debug=True);



