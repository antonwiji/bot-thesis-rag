from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek

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
