import json
import logging
import os
from flask import Flask, request, jsonify, render_template
import rag_handler as rh
import model_handler as mh
import custom_formatter as cf

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger().handlers[0].setFormatter(cf.CustomFormatter())

# Load configuration
with open("config.json", mode="r", encoding="utf-8") as read_file:
    config = json.load(read_file)

# Load local configuration overrides if exists
if os.path.exists("config.local.json"):
    with open("config.local.json", mode="r", encoding="utf-8") as read_file:
        local_config = json.load(read_file)
        for key, value in local_config.items():
            if key in config and isinstance(config[key], dict) and isinstance(value, dict):
                config[key].update(value)
            else:
                config[key] = value

# Initialize model and RAG handlers
model_handler = mh.ModelHandler(config)
rag_handler = rh.RAGHandler(config)

# Load the model
model = model_handler.load_model()
if not model:
    logging.error("Error loading model. Make sure you have installed the model and Ollama is running. Exiting...")
    exit(1)
if config["rag_options"]["clear_database_on_start"] and rag_handler.vector_store._collection.count() > 0:
    rag_handler.vector_store.reset_collection()

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    user_input = data.get("question", "")
    if not user_input:
        return jsonify({"error": "No question provided."}), 400

    if user_input == "list collections":
        collections = rag_handler.list_collections()
        response = "Available Collections:\n"
        for idx, coll in enumerate(collections):
            response += f"- {idx+1}. {coll.name}\n"
        return jsonify({"response": response, "done_reason": "stop", "total_tokens": 0})

    if user_input.startswith("switch collection to "):
        collections = rag_handler.list_collections()
        collection_idx = user_input.replace("switch collection to ", "").strip()
        if not collection_idx.isdigit() or int(collection_idx) < 1 or int(collection_idx) > len(collections):
            return jsonify({"error": "Invalid collection index."}), 400

        new_collection_name = collections[int(collection_idx)-1].name
        rag_handler.change_collection(new_collection_name)
        return jsonify({"response": f"Switched to \"{new_collection_name}\"", "done_reason": "stop", "total_tokens": 0})

    if rag_handler.vector_store._collection.count() > 0:
        related_docs = rag_handler.get_docs_by_similarity(user_input)
        response = model_handler.get_response(user_input, related_docs, True)
    else:
        response = model_handler.get_response(user_input, None, False)

    return jsonify({
        "response": response.content,
        "done_reason": response.response_metadata.get("done_reason"),
        "total_tokens": response.response_metadata.get("total_tokens")
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)