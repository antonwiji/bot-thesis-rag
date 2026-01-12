from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate

from llm_deepseek import get_deepseek_llm
from llm_deepseek import get_gpt5_nano_llm

PERSIST_DIR = "./chroma_laptop_db"
COLLECTION_NAME = "laptop_rag"

load_dotenv()


def get_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    return retriever


def build_rag_chain():
    retriever = get_retriever()
    llm = get_deepseek_llm()

    template = """
Kamu adalah asisten yang menjawab berdasarkan informasi produk laptop.

Gunakan hanya KONTEN berikut sebagai konteks:
{context}

Pertanyaan pengguna:
{question}

Berikan jawaban yang:
- spesifik berdasarkan title, deskripsi, dan harga produk
- jangan mengarang jika data tidak cukup (jawab: "maaf, datanya belum cukup")

Jawaban:
"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"],
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

    return qa_chain

def build_rag_chain_gpt():
    retriever = get_retriever()
    llm = get_gpt5_nano_llm()

    template = """
Kamu adalah asisten yang menjawab berdasarkan informasi produk laptop.

Gunakan hanya KONTEN berikut sebagai konteks:
{context}

Pertanyaan pengguna:
{question}

Berikan jawaban yang:
- spesifik berdasarkan title, deskripsi, dan harga produk
- jangan mengarang jika data tidak cukup (jawab: "maaf, datanya belum cukup")

Jawaban:
"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"],
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

    return qa_chain


if __name__ == "__main__":
    qa = build_rag_chain()
    q = "Laptop apa yang cocok untuk gaming dengan budget 15 juta?"
    res = qa.invoke({"query": q})
    print("Q:", q)
    print("A:", res["result"])
    print("\nSumber:")
    for d in res["source_documents"]:
        print(
            "-",
            d.metadata.get("product_id"),
            "|",
            d.metadata.get("title"),
            "|",
            d.metadata.get("price"),
        )
