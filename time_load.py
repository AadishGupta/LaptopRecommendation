import time, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

print("Testing pkl load time...")
t = time.time()
import pickle
with open("data/index.pkl", "rb") as f:
    records = pickle.load(f)
print(f"  pkl load: {time.time()-t:.2f}s  ({len(records)} records)")

print("Testing faiss load time...")
t = time.time()
import faiss
index = faiss.read_index("data/index.faiss")
print(f"  faiss load: {time.time()-t:.2f}s  ({index.ntotal} vectors)")

print("Testing single Ollama embed...")
t = time.time()
import ollama
resp = ollama.embed(model="nomic-embed-text", input="test laptop")
print(f"  embed: {time.time()-t:.2f}s")

print("Testing single Ollama chat...")
t = time.time()
resp = ollama.chat(model="llama3.1", messages=[{"role":"user","content":"say hi"}])
print(f"  chat: {time.time()-t:.2f}s")
