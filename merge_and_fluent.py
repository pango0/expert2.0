from langchain_community.llms.ollama import Ollama
from langchain.prompts import ChatPromptTemplate
from get_llm import get_llm

PROMPT_TEMPLATE = """
融合下列兩份文章
第一份:
{first}
第二份:
{second}
"""

def merge(response_1, response_2):
    model = get_llm()
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(first = response_1, second = response_2)
    response_text = model.invoke(prompt)
    return response_text