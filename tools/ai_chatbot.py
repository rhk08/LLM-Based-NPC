import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_cohere import ChatCohere
from langchain_chroma import Chroma
from langchain_cohere import CohereEmbeddings
import chromadb

from langchain_text_splitters import RecursiveCharacterTextSplitter
from uuid import uuid4

import shutil

class AIChatbot:
    def __init__(self, api_key_file='api.txt', game="elden_ring", character="white_mask_varre"):
        # Read the API key from file
        self.api_key = self._read_api_key(api_key_file)
        # Initialize the ChatCohere object with the API key
        self.chat = ChatCohere(cohere_api_key=self.api_key)
        
        # Set names
        self.game = game
        self.character = character
        
        # Check game knowledge dir
        game_vectordb_dir = f"{game}/public_knowledge"
        if self._vector_db_exists(game_vectordb_dir):
            print(f"{game} vector database exists. Loading it...")
            self.game_knowledge_vectordb = Chroma(
                collection_name="public_knowledge",
                embedding_function=self._initialize_embeddings(),
                persist_directory=game_vectordb_dir
            )
        else:
            print(f"{game} vector database not found. Creating a new one, this may take a while...")
            
            #Read .txt file for info
            file_path = f'{game}/public_knowledge.txt'
            if os.path.exists(file_path):
                with open(file_path, errors='ignore') as f:
                    public_knowledge = f.read()
                    
                if not public_knowledge.strip():
                    raise ValueError(f"Error: The file '{file_path}' is empty.")
            else:
                print(f"Error: The file '{file_path}' does not exist.")
            
            #Split document into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size = 1500,
                chunk_overlap  = 500,
                length_function = len,
                separators = ['\n\n\n', '\n\n', '\n', '.', ',', ' '],
                is_separator_regex=False,
            )
            documents = text_splitter.create_documents([public_knowledge])
            uuids = [str(uuid4()) for _ in range(len(documents))]
            
            #Create the new database
            self.game_knowledge_vectordb = Chroma(
                collection_name="public_knowledge",
                embedding_function=self._initialize_embeddings(),
                persist_directory=game_vectordb_dir
            )
            
            #Add the new split documents
            self.game_knowledge_vectordb.add_documents(documents=documents, ids=uuids)
        
        # Check character knowledge dir
        character_vectordb_dir = f"{game}/characters/{character}/character_knowledge"
        if self._vector_db_exists(character_vectordb_dir):
            print(f"{character} database exists, Loading it...")
            self.character_knowledge_vectordb = Chroma(
                collection_name=f"{character}_knowledge",
                embedding_function=self._initialize_embeddings(),
                persist_directory=character_vectordb_dir
            )
        else:
            print(f"{character} vector database not found. Creating a new one, this may take a while...")
            
            #Read .txt file for info
            file_path = f'{game}/characters/{character}/character_knowledge.txt'
            if os.path.exists(file_path):
                with open(file_path, errors='ignore') as f:
                    character_knowledge = f.read()
                    
                if not character_knowledge.strip():
                    raise ValueError(f"Error: The file '{file_path}' is empty.")
            else:
                print(f"Error: The file '{file_path}' does not exist.")
            
            #Split document into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size = 1500,
                chunk_overlap  = 500,
                length_function = len,
                separators = ['\n\n\n', '\n\n', '\n', '.', ',', ' '],
                is_separator_regex=False,
            )
            documents = text_splitter.create_documents([character_knowledge])
            uuids = [str(uuid4()) for _ in range(len(documents))]
            
            #Create the new database
            self.character_knowledge_vectordb = Chroma(
                collection_name=f"{character}_knowledge",
                embedding_function=self._initialize_embeddings(),
                persist_directory=character_vectordb_dir
            )
            
            #Add the new split documents
            self.character_knowledge_vectordb.add_documents(documents=documents, ids=uuids)
    
    def _read_api_key(self, api_key_file):
        """Read the API key from a file."""
        try:
            with open(api_key_file, errors='ignore') as f:
                return f.read().strip()
        except FileNotFoundError:
            print(f"Error: API key file '{api_key_file}' not found.")
            return None
    
    def _initialize_embeddings(self):
        """Initialize the Cohere embeddings."""
        return CohereEmbeddings(cohere_api_key=self.api_key, model="embed-english-v3.0", user_agent='langchain')
        
    def _vector_db_exists(self, persist_directory):
        """Check if the vector database directory exists."""
        return os.path.exists(persist_directory) and os.path.isdir(persist_directory)

    def ask(self, user_input):
        """Invoke the AI model and return its response to the user's input."""
        
        
        #Info to add
        # - World Setting
        
        
        # - Character Ai is Roleplaying
        # - Character Biography
        # - Characters talking style
        
        
        # Additional information
        #  - Public World Knowledge
        #  - Character Specific Knowledge
        
        
        # Preconversation
        #  - Preconversation
            # Relevent Conversation
            # Summarized Conversation
            # Basic Conversation
        
        
        # Instructions for response.
        
        
        # Set up the system and user message templates
        self.prompt_template = ChatPromptTemplate([
            ("system", 
             '''You are a helpful assistant called {character}.
             '''),
            ("user", "{user_input}")
        ])

        # Chain the prompt template and chat model
        self.chain = self.prompt_template | self.chat
        
        response = self.chain.invoke({"user_input": user_input, "character": self.ai_character})
        return response.content if response else None
    
    def tokenize_word(self, text):
        """Tokenize a given word using the Cohere API and return the number of tokens."""
        print(f"Tokenizing word: {text}...")
        try:
            # Tokenize the text using the Cohere API
            tokenized_output = self.cohere_client.tokenize(text=text, model="command-r-08-2024", offline=False)
            num_tokens = len(tokenized_output.tokens)
            print(f"Tokenization complete. Number of tokens: {num_tokens}")
            return num_tokens
        except Exception as e:
            print(f"Error during tokenization: {e}")
            return None
