# Ollama related
import sys
import requests
import json
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama.chat_models import ChatOllama
from langchain.callbacks.base import BaseCallbackHandler

class ModelHandler:
    def __init__(self, config):
        self.config = config
        
        self.model = self.load_model()
        
        self.prompt_template = PromptTemplate(
            input_variables=["context", "user_input"],
            template=self.config["llm_options"]["user_prompt"]
        )

        self.chat_prompt = ChatPromptTemplate.from_messages([
            ("system", self.config["llm_options"]["system_prompt"]),
            MessagesPlaceholder("conversation"),  # Dynamic history insertion
            ("human", "{current_question}")
        ])

        self.conversation_history = []
    
    # Load the Ollama model
    def load_model(self):
        try:
            return ChatOllama(
                    model=self.config["llm_options"]["model"], 
                    validate_model_on_init=True,
                    base_url=self.config["llm_options"]["ollama_address"],
                    temperature=self.config["llm_options"]["temperature"],
                    num_predict=self.config["llm_options"]["tokens_to_generate"],
                )
        except Exception as e:
            print(f"Error loading model: {e}\n Make sure you have installed the model and ollama is running")
            exit(1)

    # Combine the contents of related documents into a single context string
    def combine_context(self, related_docs):
        context = ""
        for result in related_docs:
            # Add the content of each document part to the context
            context += result.page_content+"\n"
        return context

    # Get response from the model
    def get_response(self, user_input, related_docs, useRAG=False):
        if useRAG:
            # Combine the contents of the related document parts into a single context string
            context = self.combine_context(related_docs)
            prompt = self.prompt_template.format(context=context, user_input=user_input)
        else:
            prompt = user_input

        # Format messages for the chat model
        formatted_messages = self.chat_prompt.format_messages(
            conversation=self.config["llm_options"]["use_short_term_memory"] and self.conversation_history or [],
            current_question=prompt,
        )

        if self.config["llm_options"]["use_short_term_memory"]:
            if sys.getsizeof(formatted_messages) > self.config["llm_options"]["max_context_size"]:
                print("Warning: The conversation history is too large. Deleteing oldest messages to fit the context size.")
                
                self.conversation_history = self.conversation_history[1:] # Remove the oldest message

                # Reformat messages after trimming history
                formatted_messages = self.chat_prompt.format_messages(
                    conversation=self.conversation_history,
                    current_question=prompt,
                )

        # Get the response from the model
        response = self.model.invoke(formatted_messages)

        # If short-term memory is enabled, store the interaction
        if self.config["llm_options"]["use_short_term_memory"]:
            self.conversation_history.append(HumanMessage(content=prompt))
            self.conversation_history.append(response)

            

        

        return response