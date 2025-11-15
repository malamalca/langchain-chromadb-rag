import logging, json, os, sys, argparse, re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import chromadb
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
import custom_formatter as cf
import custom_text_splitter

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

length_function = len

# Command line arguments
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("-h", "--help", action="help",
            help="To run this script please provide input text file as an argument.\n")
parser.add_argument("txt_file", type=str, nargs="?",
            help="Path to the input text file to be ingested into the vector database.\n")
parser.add_argument("--dry-run", action="store_true", default=False,
            help="If set, the script will only print the splits without adding them to the database.\n")
parser.add_argument("--collection-name", type=str, default="information",
            help="Name of the collection in the vector database.\n")
args, unknown = parser.parse_known_args()

if not args.txt_file or not os.path.exists(args.txt_file):
    logging.error(f"Input file parameter does not exist.")
    sys.exit(1)

# The default list of split characters is [\n\n, \n, " ", ""]
# Tries to split on them in order until the chunks are small enough
# Keep paragraphs, sentences, words together as long as possible
# deli po: Prvi del, II. poglavje, 1.1. naslov, 23. člen, konec odstavka, konec stavka, presledek
splitter = RecursiveCharacterTextSplitter(
    separators=[r".* del: .*\n", r"^[I,V,X]{1,2}\. poglavje.*\n", r"\n\d\.(?:(?:\d{1,3}\.)?\d{1,3})?.*\n", r"\n(?:\ *)?\d{1,4}\.(?:.*)? člen\n", r"\n", r" ", r""],
    is_separator_regex=True,
    chunk_size=config["splitter_options"]["chunk_size"], 
    chunk_overlap=config["splitter_options"]["chunk_overlap"],
    length_function=length_function,
)

# Load and split the document
loader = TextLoader(args.txt_file, encoding="utf-8")
documents = loader.load()
chunks = splitter.split_documents(documents)

max_len = max([length_function(s.page_content) for s in chunks])

# Add lowercase source file metadata to each chunk
current_article_part_no = 1
current_article_no = None
for chunk in chunks:
    match = re.search(r"^(?:\ *)?\d{1,4}\.(?:.*)? člen\n", chunk.page_content)
    if match:
        current_article_part_no = 1
        current_article_no = match.group(0).split(". člen")[0].strip()
        # Remove the article title from the content
        lines = chunk.page_content.splitlines()
        chunk.page_content = '\n'.join(lines[1:])

    if not match and current_article_no:
        current_article_part_no += 1


    chunk.page_content = f"člen številka: {current_article_no}\ndel člena: {current_article_part_no}\n\n{chunk.page_content}"

    chunk.metadata["original_text"] = chunk.page_content
    chunk.metadata["clen"] = current_article_no
    chunk.page_content = chunk.page_content.lower()
    chunk.metadata["source_file"] = os.path.basename(args.txt_file).lower()

if not args.dry_run:
    logging.info(f"Adding document to the database, collection: {args.collection_name}")
    vector_store = Chroma(
        collection_name=args.collection_name,
        persist_directory=config["rag_options"]["database_folder"],
        #embedding_function=FastEmbedEmbeddings(),
        #embedding_function=OllamaEmbeddings(model="nomic-embed-text",base_url=config["llm_options"]["ollama_address"]),
        embedding_function=OllamaEmbeddings(model="bge-m3",base_url=config["llm_options"]["ollama_address"]),

        client_settings=chromadb.config.Settings(
            anonymized_telemetry=False,
        ),
    )

    vector_store.add_documents(chunks)
    logging.info(f"Added document to the database.")

for i, split in enumerate(chunks):
    logging.warning(f"--- Split {i+1} ---")
    print(split.page_content)
    print()

logging.info(f"Number of splits: {len(chunks)}")
logging.info(f"Max split length: {max_len}")
