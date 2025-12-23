import os
import shutil
from typing import List

import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from models import Product

PERSIST_DIR = "./chroma_laptop_db"
COLLECTION_NAME = "laptop_rag"
CSV_PATH = "data/product_semicolon.csv" 

def load_products_csv(path: str) -> List[Product]:
    """
    Load data produk dari CSV dengan kolom:
    title, price, description.
    id digenerate dari nomor baris (1,2,3,...).
    """
    df = pd.read_csv(path, sep=";")

    products: List[Product] = []
    for idx, row in df.iterrows():
        products.append(
            Product(
                id=str(idx + 1),  # id = nomor baris (1-based)
                title=str(row.get("title", "")),
                price=str(row.get("price", "")),
                description=str(row.get("description", "")),
            )
        )

    return products


def main():
    # 1. Load produk dari CSV
    products = load_products_csv(CSV_PATH)
    print(f"Loaded {len(products)} products from CSV")

    # 2. Hapus folder Chroma lama (jika ada) untuk reset total
    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)
        print(f"🧹 Menghapus folder lama: {PERSIST_DIR}")

    # 3. Siapkan embeddings lokal (tanpa OpenAI)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 4. Inisialisasi Chroma dengan persistence
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )

    # 5. Siapkan teks + metadata
    texts = [
        f"{p.title}. Deskripsi: {p.description}. Harga: {p.price}"
        for p in products
    ]

    metadatas = [
        {
            "product_id": p.id,       # dipakai untuk evaluasi
            "title": p.title,
            "price": p.price,
        }
        for p in products
    ]

    ids = [p.id for p in products]

    # 6. Masukkan ke Chroma (auto-persist, TIDAK perlu .persist())
    vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)

    print(f"✅ Selesai ingest {len(products)} produk ke Chroma dari CSV.")
    print(f"   Folder DB: {PERSIST_DIR}")


if __name__ == "__main__":
    main()
