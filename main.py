import create_database
import argparse
import summarize
import summary_database
import time
from functools import wraps
from merge_and_fluent import merge
from get_llm import get_llm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

def parse_arguments():
    parser = argparse.ArgumentParser(description="A script with mutually exclusive -q and -s flags")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-q", "--query", action="store_true", help="Ask questions")
    group.add_argument("-s", "--summarize", action="store_true", help="Summarize data")
    group.add_argument("-c", "--create_database", action="store_true", help="Create vector database")
    return parser.parse_args()

def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"{func.__name__} took {execution_time:.4f} seconds to execute.")
        return result
    return wrapper



PROMPT_TEMPLATE = """
你是一位以下內容的專家，僅根據以下提供的內容用markdown格式回答問題：

{context}

---

僅根據以上提供的內容回答問題，如果問題與內容無關就回答"此問題我無法回答"就好，不要回覆額外內容: {question}
"""

@timing_decorator
def query_rag(query_text: str, db_path: str, top):
        embedding_function = create_database.get_embedding_function()
        db = Chroma(persist_directory=db_path, embedding_function=embedding_function)
        results = db.similarity_search_with_score(query_text, k=top)
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)

        model = get_llm()
        response_text = model.invoke(prompt)
        formatted_response = f"Response: {response_text}"
        # print("\n"+formatted_response)
        return response_text
    
@timing_decorator
def main():

    args = parse_arguments()

    if args.query:
        query = input("向專家發問:")

        ret1 = query_rag(query, "chunk_150_50", 20)
        ret2 = query_rag(query, "chunk_500_150", 10)
        ret3 = query_rag(query, "summary_db", 10)

        print("Final:")
        print(merge(merge(ret1, ret2), ret3))
    elif args.summarize:
        summarize.summarize("data")
    elif args.create_database:
        create_database.create_database("data", "chunk_150_50", 150, 50)
        create_database.create_database("data", "chunk_500_150", 500, 150) # 400 char
        summary_database.create_database("summaries", "summary_db", 500, 150)
    else:
        summarize.summarize("data")
        create_database.create_database("data", "chunk_150_50", 150, 50)
        create_database.create_database("data", "chunk_500_150", 500, 150) # 400 char
        summary_database.create_database("summaries", "summary_db", 500, 150)
        query = input("向專家發問:")

        ret1 = query_rag(query, "chunk_150_50", 20)
        ret2 = query_rag(query, "chunk_500_150", 10)
        ret3 = query_rag(query, "summary_db", 10)

        print("Final:")
        print(merge(merge(ret1, ret2), ret3))
    


if __name__ == "__main__":
    main()
    