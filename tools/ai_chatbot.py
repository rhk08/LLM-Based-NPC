from langchain_core.prompts import ChatPromptTemplate
from langchain_cohere import ChatCohere

class AIChatbot:
    def __init__(self, api_key_file='api.txt', assistant_name="John"):
        # Read the API key from file
        self.api_key = self._read_api_key(api_key_file)

        # Initialize the ChatCohere object with the API key
        self.chat = ChatCohere(cohere_api_key=self.api_key)

        # Set up the system and user message templates
        self.prompt_template = ChatPromptTemplate([
            ("system", f"You are a helpful assistant called {assistant_name}."),
            ("user", "{user_input}")
        ])

        # Chain the prompt template and chat model
        self.chain = self.prompt_template | self.chat

    def _read_api_key(self, api_key_file):
        """Read the API key from a file."""
        try:
            with open(api_key_file, errors='ignore') as f:
                return f.read().strip()
        except FileNotFoundError:
            print(f"Error: API key file '{api_key_file}' not found.")
            return None

    def ask(self, user_input):
        """Invoke the AI model and return its response to the user's input."""
        response = self.chain.invoke({"user_input": user_input})
        return response.content if response else None
