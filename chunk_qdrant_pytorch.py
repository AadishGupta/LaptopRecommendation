# chunk_qdrant_pytorch.py - PyTorch version with consistent GPU usage

import os
import json
import time
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer

# =============================================================================
# FORCE GPU USAGE
# =============================================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# =============================================================================
# CONFIGURATION
# =============================================================================
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
BATCH_SIZE = 256  # Good balance for RTX 3050
SOURCE_COLLECTION = "laptops"  # raw, unchunked catalog (one point per laptop)
COLLECTION_NAME = "laptops_chunked"
OUTPUT_DIR = "data/chunked_qdrant/"

print("="*60)
print("🚀 PYTORCH GPU CHUNKING (CONSISTENT GPU)")
print("="*60)

# =============================================================================
# CHECK GPU
# =============================================================================
print(f"\n[GPU] CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[GPU] Device: {torch.cuda.get_device_name(0)}")
    print(f"[GPU] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    # Force GPU warm-up
    _ = torch.ones(1000, 1000).cuda()
    print("[GPU] ✅ GPU is ready!")
else:
    print("[GPU] ❌ GPU not available - using CPU (will be slower)")
    exit(1)

start_time = time.time()

# =============================================================================
# LOAD MODEL ON GPU
# =============================================================================
print("\n[Loading] Loading embedding model on GPU...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer('BAAI/bge-small-en-v1.5', device=device)
# FIXED: Just print the device directly
print(f"   ✅ Model loaded on: {model.device}")

# =============================================================================
# CONNECT TO QDRANT
# =============================================================================
print("\n[1/4] Connecting to Qdrant...")
try:
    client = QdrantClient(host="localhost", port=6333)
    client.get_collections()
    print("   ✅ Connected to Qdrant")
except Exception as e:
    print(f"   ❌ Failed to connect: {e}")
    print("   Make sure Qdrant is running: qdrant --config-path config.yaml")
    exit(1)

# =============================================================================
# LOAD DATA
# =============================================================================
print("\n[2/4] Loading laptops from Qdrant...")


def _load_raw_laptops(client: QdrantClient, collection_name: str) -> list:
    """Scroll every point out of the raw (unchunked) laptops collection and
    return it as a plain list of {id, name, price, description} dicts —
    the same shape the rest of this script previously got from the pkl
    file. Requires SOURCE_COLLECTION to already exist (e.g. populated by
    upload_to_qdrant.py) before this script is run."""
    records = []
    next_offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=collection_name,
            limit=256,
            offset=next_offset,
            with_payload=True,
            with_vectors=False,
        )
        for p in points:
            payload = p.payload or {}
            records.append({
                "id": payload.get("id", p.id),
                "name": payload.get("name", ""),
                "price": payload.get("price", 0),
                "description": payload.get("description", ""),
            })
        if next_offset is None:
            break
    return records


try:
    original_laptops = _load_raw_laptops(client, SOURCE_COLLECTION)
except Exception as e:
    print(f"   ❌ Failed to load laptops from Qdrant collection '{SOURCE_COLLECTION}': {e}")
    print(f"   Make sure '{SOURCE_COLLECTION}' exists and is populated (e.g. via upload_to_qdrant.py).")
    exit(1)

if not original_laptops:
    print(f"   ❌ No laptops found in Qdrant collection '{SOURCE_COLLECTION}'")
    exit(1)

print(f"   Loaded {len(original_laptops):,} laptops")

# =============================================================================
# CHUNK
# =============================================================================
print("\n[3/4] Chunking laptops...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
)

chunked_data = []
all_chunk_texts = []
chunk_to_laptop = {}

for idx, laptop in enumerate(tqdm(original_laptops, desc="Chunking", unit="laptops")):
    desc = laptop.get("description", "")
    chunks = text_splitter.split_text(desc) or [desc]
    
    for chunk_idx, chunk_text in enumerate(chunks):
        chunk_id = f"{laptop['id']}_{chunk_idx}"
        chunked_data.append({
            "id": chunk_id,
            "laptop_id": laptop["id"],
            "name": laptop["name"],
            "price": laptop["price"],
            "description": chunk_text,
            "full_description": desc,
            "chunk_index": chunk_idx,
            "total_chunks": len(chunks),
        })
        all_chunk_texts.append(chunk_text)
        chunk_to_laptop[chunk_id] = laptop["id"]

print(f"   Created {len(chunked_data):,} chunks")

# =============================================================================
# EMBED WITH PYTORCH (CONSISTENT GPU)
# =============================================================================
print(f"\n[4/4] Embedding with PyTorch (GPU)...")

embeddings = []
total = len(all_chunk_texts)

# Set model to evaluation mode for faster inference
model.eval()

with tqdm(total=total, desc="Embedding GPU", unit="chunks") as pbar:
    for i in range(0, total, BATCH_SIZE):
        batch = all_chunk_texts[i:i+BATCH_SIZE]
        
        # Encode batch on GPU
        with torch.no_grad():
            batch_embeddings = model.encode(
                batch,
                convert_to_numpy=True,
                device='cuda',
                show_progress_bar=False,
                normalize_embeddings=True,
            )
        embeddings.extend(batch_embeddings)
        
        pbar.update(len(batch))
        
        # Show GPU memory usage periodically
        if i % (BATCH_SIZE * 5) == 0 and i > 0 and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            print(f"\n   💾 VRAM Used: {allocated:.2f} GB")

print(f"   Generated {len(embeddings):,} embeddings")

# =============================================================================
# UPLOAD TO QDRANT
# =============================================================================
print("\n[Uploading] Sending to Qdrant...")

try:
    client.delete_collection(COLLECTION_NAME)
    print("   Removed existing collection")
except:
    pass

VECTOR_SIZE = len(embeddings[0])
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
)
print(f"   Collection created (dim={VECTOR_SIZE})")

# Upload in batches
points = []
embeddings_np = np.array(embeddings, dtype=np.float32)

with tqdm(total=len(chunked_data), desc="Uploading", unit="chunks") as pbar:
    for idx, chunk in enumerate(chunked_data):
        points.append(PointStruct(
            id=idx,
            vector=embeddings_np[idx].tolist(),
            payload={
                "chunk_id": chunk["id"],
                "laptop_id": chunk["laptop_id"],
                "name": chunk["name"],
                "price": chunk["price"],
                "description": chunk["description"],
                "full_description": chunk["full_description"],
            }
        ))
        
        if len(points) >= BATCH_SIZE or idx == len(chunked_data) - 1:
            client.upsert(collection_name=COLLECTION_NAME, points=points)
            points = []
            pbar.update(len(points) if idx == len(chunked_data) - 1 else BATCH_SIZE)

print(f"   ✅ Uploaded {len(chunked_data):,} chunks")

# =============================================================================
# SAVE METADATA
# =============================================================================
print("\n[Saving] Writing metadata...")
os.makedirs(OUTPUT_DIR, exist_ok=True)

metadata = {
    "total_laptops": len(original_laptops),
    "total_chunks": len(chunked_data),
    "chunk_size": CHUNK_SIZE,
    "chunk_overlap": CHUNK_OVERLAP,
    "batch_size": BATCH_SIZE,
    "vector_size": VECTOR_SIZE,
    "collection_name": COLLECTION_NAME,
    "embedding_model": "BAAI/bge-small-en-v1.5",
    "device": device,
    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    "created_at": time.ctime(),
    "build_time_seconds": time.time() - start_time,
}

with open(f"{OUTPUT_DIR}metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

with open(f"{OUTPUT_DIR}chunk_to_laptop.json", "w") as f:
    json.dump(chunk_to_laptop, f)

print(f"   ✅ Saved metadata")

# =============================================================================
# TEST SEARCH
# =============================================================================
print("\n[Testing] Search verification...")

test_embedding = model.encode(
    ["gaming laptop with RTX 4080"],
    convert_to_numpy=True,
    device='cuda',
)

search_results = client.search(
    collection_name=COLLECTION_NAME,
    query_vector=test_embedding[0].tolist(),
    limit=5,
    with_payload=True,
)

print("\n   📊 Top 5 results:")
for i, result in enumerate(search_results, 1):
    payload = result.payload
    print(f"   {i}. {payload['name']} - ₹{payload['price']:,} (score: {result.score:.3f})")

# =============================================================================
# COMPLETE
# =============================================================================
elapsed = time.time() - start_time
print("\n" + "="*60)
print("✅ COMPLETE!")
print(f"   Time: {elapsed/60:.1f} minutes")
print(f"   Chunks: {len(chunked_data):,}")
print(f"   Device: {device.upper()}")
if torch.cuda.is_available():
    print(f"   VRAM Used: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
print("="*60)

print("\n🚀 NEXT STEPS:")
print("   1. Keep Qdrant running")
print("   2. Run: python agent_app.py")