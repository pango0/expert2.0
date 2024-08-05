from langchain_community.llms.ollama import Ollama
def get_llm(model_name = "cwchang/llama3-taide-lx-8b-chat-alpha1:latest"):
    return Ollama(model=model_name)