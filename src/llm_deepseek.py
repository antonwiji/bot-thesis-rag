from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
import os

# Load .env (DEEPSEEK_API_KEY)
load_dotenv()

def get_deepseek_llm():
    """
    Mengembalikan instance LLM DeepSeek untuk dipakai di RAG chain.
    Pastikan env var DEEPSEEK_API_KEY sudah diset di .env atau environment.
    """
    llm = ChatDeepSeek(
        model="deepseek-chat",  # bisa diganti model lain yang kamu pakai
        # api_key akan otomatis diambil dari DEEPSEEK_API_KEY
    )
    return llm

def get_gpt5_nano_llm():
    """
    Mengembalikan instance LLM OpenAI (GPT 5 nano) untuk dipakai di RAG chain.
    Pastikan OPENAI_API_KEY sudah diset di .env atau environment.
    """
    llm = ChatOpenAI(
        model="gpt-5-nano",        # ganti sesuai nama model di akunmu kalau beda
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    return llm