import json
import logging
# For directory watcher
import os
import watchdog
from watchdog.observers import Observer

import rag_handler as rh
import model_handler as mh
import custom_formatter as cf


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger().handlers[0].setFormatter(cf.CustomFormatter())

# Load configuration
with open("config.json", mode="r", encoding="utf-8") as read_file:
    global config
    config = json.load(read_file)
    read_file.close()

# Load local configuration overrides if exists
if os.path.exists("config.local.json"):
    with open("config.local.json", mode="r", encoding="utf-8") as read_file:
        local_config = json.load(read_file)
        for key, value in local_config.items():
            if key in config and isinstance(config[key], dict) and isinstance(value, dict):
                config[key].update(value)
            else:
                config[key] = value
        read_file.close()

# Watch the ingestion folder
class FileSystemWatcher(watchdog.events.FileSystemEventHandler):
    def on_created(self, event):
        file_name = event.src_path.split('/')[-1]
        if not file_name.startswith('.'):  # Ignore hidden files
            logging.info(f"Detected and Ingesting: {event.src_path}")
            docs = rag_handler.load_document(event.src_path)
            # Delete the file after ingestion if thew option is true in config
            if config["rag_options"]["delete_file_after_ingestion"] and os.path.exists(event.src_path):
                os.remove(event.src_path)
                logging.info(f"Deleted After Ingestion: {event.src_path}")
            # Add the documents to the database
            rag_handler.add_document_to_chroma(docs)
            logging.info(f"Ingested: {event.src_path}")
    def on_deleted(self, event):
        logging.info(f"Deleted: {event.src_path}")

# Start the folder observer
observer = Observer()
observer.schedule(FileSystemWatcher(), path=config["rag_options"]["ingestion_folder"], recursive=True)
observer.start()

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

# Main loop
def main():
    print("Welcome, type 'help' for help and 'exit' to exit.")
    try:
        while True:
            user_input = input(">> ")
            if user_input == "exit":
                observer.stop()
                print("Goodbye!")
                logging.info("Exiting...")
                break
            if user_input == "help":
                print(parser.format_help())
                continue
            if user_input == "clear":
                model_handler.conversation_history.clear()
                print("Database cleared.")
                continue
            # Use RAG if chromadb exists, otherwise, just use the model
            if rag_handler.vector_store._collection.count() > 0:
                related_docs = rag_handler.get_docs_by_similarity(user_input)
                logging.info(f"Related docs: {len(related_docs)}")  # Debug
                response = model_handler.get_response(user_input, related_docs, True)
            else:
                logging.warning("No documents in database, using only the model.")  # Debug
                response = model_handler.get_response(user_input, None, False)
            
            logging.debug(f"Done Reason: {response.response_metadata.get('done_reason')}")  # Debug
            logging.debug(f"Token Count: {response.response_metadata.get('total_tokens')}")  # Debug

            print(f"Response: \n{cf.yellow}{response.content}{cf.reset}")
    except KeyboardInterrupt:
        observer.stop()
        logging.info("Exiting...")
    observer.join()

if __name__ == '__main__':
    main()