"""
upload_to_qdrant.py  —  run once to upload laptop vectors to Qdrant
Make sure Qdrant is running first:
    docker run -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant
"""
import pickle, sys
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct, PayloadSchemaType
)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PKL_PATH        = "data/index.pkl"
COLLECTION_NAME = "laptops"
BATCH_SIZE      = 200

# ── Load pkl ───────────────────────────────────────────────────────────────
print("Loading pkl...")
with open(PKL_PATH, "rb") as f:
    records = pickle.load(f)
print(f"  {len(records)} laptops")

dim = len(records[0]["full_embedding"])
print(f"  Embedding dim: {dim}")

# ── Connect to Qdrant ──────────────────────────────────────────────────────
print("\nConnecting to Qdrant at localhost:6333...")
client = QdrantClient(host="localhost", port=6333)

# ── Create collection (drop if exists) ────────────────────────────────────
existing = [c.name for c in client.get_collections().collections]
if COLLECTION_NAME in existing:
    print(f"  Dropping existing collection '{COLLECTION_NAME}'...")
    client.delete_collection(COLLECTION_NAME)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
)
print(f"  Collection '{COLLECTION_NAME}' created (dim={dim}, cosine)")

# Create payload index on price for fast filtered search
client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="price",
    field_schema=PayloadSchemaType.INTEGER,
)
print("  Payload index on 'price' created")

# ── Upload in batches ──────────────────────────────────────────────────────
print(f"\nUploading {len(records)} laptops in batches of {BATCH_SIZE}...")
total = 0
for i in range(0, len(records), BATCH_SIZE):
    batch = records[i:i+BATCH_SIZE]
    points = [
        PointStruct(
            id=rec["id"],
            vector=rec["full_embedding"],
            payload={
                "name":        rec["name"],
                "description": rec["description"],
                "price":       rec["price"],
                "chunks":      rec["chunks"],
            }
        )
        for rec in batch
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    total += len(batch)
    print(f"  Uploaded {total}/{len(records)}")

print(f"\nDone. {total} laptops in Qdrant collection '{COLLECTION_NAME}'.")
info = client.get_collection(COLLECTION_NAME)
print(f"  Vectors count: {info.vectors_count}")
