from sentence_transformers import SentenceTransformer
import sys

embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("Loaded embedder")

queries = [
    "สวัสดีค่ะ",
    "ปีาี่มีเรียนวิชาอะไรบ้าง",
    "ชั้นสามารถถามเกี่ยวกับอะไรได้บ้า",
    "ครั้งที่แล้วเราคุยอะไรกันไปบ้าง",
    "ตารางเรียนวันนี้มีอะไรบ้าง",
    "ตึกโหลคือตึกอะไร",
    "แล้วeccล่ะ",
    "ตามเล่มหลักสูตรปี1ต้องเรียนอะไรบ้าง"
]

for q in queries:
    try:
        res = embedder.encode(q, normalize_embeddings=True)
        print(f"Success: {q[:10]}... type={type(res)}")
    except Exception as e:
        print(f"Error on '{q}': {e}")
