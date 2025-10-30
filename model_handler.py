# Ollama related // Ollama ile alakalı
import sys
import requests
import json
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama.chat_models import ChatOllama

class ModelHandler:
    def __init__(self, cli_args, config):
        self.cli_args = cli_args
        self.config = config
        
        self.model = self.load_model()
        self.model.invoke([SystemMessage(cli_args.system_prompt)])
        
        self.prompt_template = PromptTemplate(
            input_variables=["context", "user_input"],
            template="""Use the following context to answer the question. 
                Context: {context}
                Question: {user_input}
            """
        )

        self.chat_prompt = ChatPromptTemplate.from_messages([
            ("system", cli_args.system_prompt),
            MessagesPlaceholder("conversation"),  # Dynamic history insertion
            ("human", "{current_question}")
        ])

        self.conversation_history = []
    
    def load_model(self):
        try:
            return ChatOllama(
                    model=self.cli_args.model, 
                    validate_model_on_init=True,
                    base_url=self.cli_args.ollama_address,
                    temperature=self.config["llm_options"]["temperature"],
                    num_predict=self.config["llm_options"]["tokens_to_generate"],
                    
                )
        except Exception as e:
            print(f"Error loading model: {e}\n Make sure you have installed the model and ollama is running")
            exit(1)

    def combine_context(self, related_docs):
        context = ""
        for result in related_docs:
            if self.config["rag_options"].get("use_reranker", False):
                doc = result
            else:
                doc = result[0]
            context += doc.page_content+"\n"
        return context

    def get_response(self, user_input, related_docs, useRAG=False):
        if useRAG:
            # Combine the contents of the related document parts // İlgili belge parçalarının içeriklerini birleştir
            context = self.combine_context(related_docs)
            prompt = self.prompt_template.format(context=context, user_input=user_input)
        else:
            prompt = user_input

        formatted_messages = self.chat_prompt.format_messages(
            conversation=self.conversation_history,
            current_question=prompt
        )

        if sys.getsizeof(formatted_messages) > self.config["llm_options"]["max_context_size"]:
            print("Warning: The conversation history is too large. Deleteing oldest messages to fit the context size.")
            self.conversation_history = self.conversation_history[1:] # Remove the oldest message

        response = self.model.invoke(formatted_messages)
        self.conversation_history.append(HumanMessage(content=prompt))
        self.conversation_history.append(response)
        return response

        #return self.model.invoke([HumanMessage(prompt)])