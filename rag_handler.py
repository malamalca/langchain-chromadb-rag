# For document loading, splitting, storing
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, csv_loader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_compressors import FlashrankRerank
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction;
from flashrank import Ranker 

from langchain_chroma import Chroma
import chromadb

class RAGHandler:
    def __init__(self, config):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config["splitter_options"]["chunk_size"],
            chunk_overlap=self.config["splitter_options"]["chunk_overlap"],
        )
        self.vector_store = self.initialize_chroma(self.config["rag_options"]["collection_name"])
        self.reranker = FlashrankRerank(
            score_threshold = self.config["rag_options"]["similarity_threshold"],
            top_n=self.config["rag_options"]["results_to_return"],
            model="ms-marco-MiniLM-L-12-v2",
        )

    def initialize_chroma(self, collection_name):
        return Chroma(
            collection_name=collection_name,
            persist_directory=self.config["rag_options"]["database_folder"],
            #embedding_function=FastEmbedEmbeddings(),
            embedding_function=OllamaEmbeddings(model="nomic-embed-text",base_url=self.config["llm_options"]["ollama_address"]),
            client_settings=chromadb.config.Settings(
                anonymized_telemetry=False,
            ),
        )
    
    def list_collections(self):
        return self.vector_store._client.list_collections()

    def change_collection(self, new_collection_name):
        self.vector_store = self.initialize_chroma(new_collection_name)

    def delete_collection(self, collection_name):
        collections = self.vector_store._client.list_collections()
        if collection_name in [coll.name for coll in collections]:
            self.vector_store._client.delete_collection(collection_name)

    # Load the document based on the file extension
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
        docs_and_scores = self.vector_store.similarity_search_with_relevance_scores(
            query=query,
            k=self.config["rag_options"]["results_to_return"],
            score_threshold=self.config["rag_options"]["similarity_threshold"],
        )

        # Extract only the documents from the (document, score) tuples
        docs_only = objects = [x[0] for x in docs_and_scores]

        # 2. do a full text query to get metadata (like source file names)
        collection = self.vector_store._client.get_collection(
            name=self.config["rag_options"]["collection_name"],
            embedding_function=OllamaEmbeddingFunction(
                model_name="nomic-embed-text",
                url=self.config["llm_options"]["ollama_address"]
            ),
        )

        fulltext_results = collection.query(
            query_texts=[query],
            where_document={"$contains": query},
            #n_results=self.config["rag_options"]["results_to_return"] * 2,
        )

        for i in range(len(fulltext_results["ids"][0])):
            doc = fulltext_results["documents"][0][i]
            new_doc = Document(page_content=doc, metadata=fulltext_results["metadatas"][0][i], id=fulltext_results["ids"][0][i])
            docs_only.append(new_doc)



        #for doc, score in docs_and_scores:
        #    print(f"Doc ID: {doc.metadata.get('source', 'N/A')}, Score: {score} Content:\n{doc.page_content[:200]}...\n")

        #for r in fulltext_hits:
        #    print(f"  {r['doc']} (fulltext_score: {r['fulltext_score']:.4f})")
        
        # If reranker is enabled, compress the documents
        if self.config["rag_options"].get("use_reranker", False) and len(docs_only) > 0:
            self.reranker.compress_documents(
                documents=docs_only,
                query=query,
            )

        return docs_only[:self.config["rag_options"]["results_to_return"]]