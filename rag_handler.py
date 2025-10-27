# For document loading, splitting, storing // Belge yükleme, bölme, saklama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, csv_loader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_compressors import FlashrankRerank
from flashrank import Ranker 
from langchain.retrievers import ContextualCompressionRetriever
from langchain_chroma import Chroma
import chromadb

class RAGHandler:
    def __init__(self, cli_args, config):
        self.cli_args = cli_args
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config["splitter_options"]["chunk_size"],
            chunk_overlap=self.config["splitter_options"]["chunk_overlap"],
        )
        self.vector_store = self.initialize_chroma()
        self.reranker = FlashrankRerank(
            score_threshold = self.config["rag_options"]["similarity_threshold"],
            top_n=self.config["rag_options"]["results_to_return"],
            model="ms-marco-MiniLM-L-12-v2",
        )

    def initialize_chroma(self):
        client_settings = chromadb.config.Settings(
            anonymized_telemetry=False,
        )
        return Chroma(
            collection_name="information",
            persist_directory=self.cli_args.database_folder,
            embedding_function=FastEmbedEmbeddings(),
            client_settings=client_settings,
        )

    # Load the document based on the file extension // Dosya uzantısına göre belgeyi yükle
    def load_document(self, file_path):
        loader = None
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        elif file_path.endswith(".csv"):
            loader = csv_loader.CSVLoader(file_path)
        elif file_path.endswith(".txt"):
            loader = TextLoader(file_path)
        else:
            print("Unsupported file type")
            return None
        return loader.load()

    def add_document_to_chroma(self, document):
        if document is None:
            print(f"Failed to load document.")
            return
        chunks = self.text_splitter.split_documents(document)
        self.vector_store.add_documents(chunks)
        print(f"Added document to the database.")

    def get_docs_by_similarity(self, query):
        if self.config["rag_options"].get("use_reranker", False):
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=self.reranker,
                base_retriever=self.vector_store.as_retriever(
                    search_type="similarity_score_threshold",
                    search_kwargs={
                        "k": self.config["rag_options"]["results_to_return"],
                        "score_threshold": self.config["rag_options"]["similarity_threshold"],
                    },
                ),
            )

            docs_and_scores = compression_retriever.invoke(query)
        else:
            docs_and_scores  = self.vector_store.similarity_search_with_relevance_scores(
                    query=query,
                    k=self.config["rag_options"]["results_to_return"],
                    score_threshold= self.config["rag_options"]["similarity_threshold"],
                )

        #print(f"Retrieved {len(docs_and_scores)} documents based on similarity.")
        #for i, score in enumerate(docs_and_scores):
        #    print(f"Document {i+1} - Similarity Score: {score}")
        return docs_and_scores