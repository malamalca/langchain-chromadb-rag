import logging, json, os, sys, argparse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import chromadb
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.document_loaders import TextLoader
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

length_function = len

# The default list of split characters is [\n\n, \n, " ", ""]
# Tries to split on them in order until the chunks are small enough
# Keep paragraphs, sentences, words together as long as possible
splitter = RecursiveCharacterTextSplitter(
    separators=[r"\n\d\.(?:(?:\d{1,3}\.)?\d{1,3})?.*\n", r".*. .*len\n", "\n\n", "\n"],
    is_separator_regex=True,
    chunk_size=config["splitter_options"]["chunk_size"], 
    chunk_overlap=config["splitter_options"]["chunk_overlap"],
    length_function=length_function,
)

parser = argparse.ArgumentParser(add_help=False)

parser.add_argument("-h", "--help", action="help",
            help="To run this script please provide input text file as an argument.\n")
parser.add_argument("txt_file", type=str, nargs="?",
            help="Path to the input text file to be ingested into the vector database.\n")
parser.add_argument("--dry-run", action="store_true", default=False,
            help="If set, the script will only print the splits without adding them to the database.\n")
args, unknown = parser.parse_known_args()

if not args.txt_file or not os.path.exists(args.txt_file):
    logging.error(f"Input file parameter does not exist.")
    sys.exit(1)

loader = TextLoader(args.txt_file, encoding="utf-8")
documents = loader.load()
chunks = splitter.split_documents(documents)

max_len = max([length_function(s.page_content) for s in chunks])

if not args.dry_run:
    vector_store = Chroma(
        collection_name="information",
        persist_directory=config["rag_options"]["database_folder"],
        embedding_function=FastEmbedEmbeddings(),
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
