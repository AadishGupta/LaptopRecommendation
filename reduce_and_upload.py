"""
reduce_and_upload.py — one-shot script:
    1. Load data/index.pkl (~10k laptops)
    2. Randomly sample TARGET_SIZE of them (default 1000)
    3. Save the reduced set to data/index_reduced.pkl (original left untouched)
    4. Connect to Qdrant, DELETE the old 'laptops' collection if it exists
    5. Create a fresh collection and upload the reduced laptops

Make sure Qdrant is running first:
    docker run -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant

Usage:
    python reduce_and_upload.py
    python reduce_and_upload.py --target 1000 --seed 42
"""
import pickle
import random
import argparse
import sys

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, PayloadSchemaType

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SRC_PKL         = "data/index.pkl"
DEST_PKL        = "data/index_reduced.pkl"
COLLECTION_NAME = "laptops"
BATCH_SIZE      = 200
SEED            = 42
TARGET_SIZE     = 1000


def reduce_dataset(src_path, dest_path, target_size, seed):
    print(f"Loading '{src_path}'...")
    with open(src_path, "rb") as f:
        records = pickle.load(f)
    print(f"  {len(records)} laptops loaded")

    if target_size >= len(records):
        print(f"  Target ({target_size}) >= dataset size ({len(records)}); using full dataset.")
        reduced = records
    else:
        rng = random.Random(seed)
        reduced = rng.sample(records, target_size)
        print(f"  Sampled {len(reduced)} laptops (seed={seed})")

    with open(dest_path, "wb") as f:
        pickle.dump(reduced, f)
    print(f"  Saved reduced set to '{dest_path}'")

    return reduced


def upload_to_qdrant(records, collection_name):
    dim = len(records[0]["full_embedding"])
    print(f"\nEmbedding dim: {dim}")

    print("Connecting to Qdrant at localhost:6333...")
    client = QdrantClient(host="localhost", port=6333)

    existing = [c.name for c in client.get_collections().collections]
    if collection_name in existing:
        print(f"  Dropping existing collection '{collection_name}'...")
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    print(f"  Collection '{collection_name}' created (dim={dim}, cosine)")

    client.create_payload_index(
        collection_name=collection_name,
        field_name="price",
        field_schema=PayloadSchemaType.INTEGER,
    )
    print("  Payload index on 'price' created")

    print(f"\nUploading {len(records)} laptops in batches of {BATCH_SIZE}...")
    total = 0
    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i:i + BATCH_SIZE]
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
        client.upsert(collection_name=collection_name, points=points)
        total += len(batch)
        print(f"  Uploaded {total}/{len(records)}")

    print(f"\nDone. {total} laptops in Qdrant collection '{collection_name}'.")
    info = client.get_collection(collection_name)
    print(f"  Vectors count: {info.vectors_count}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=SRC_PKL)
    parser.add_argument("--dest", default=DEST_PKL)
    parser.add_argument("--target", type=int, default=TARGET_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--collection", default=COLLECTION_NAME)
    args = parser.parse_args()

    reduced = reduce_dataset(args.src, args.dest, args.target, args.seed)
    upload_to_qdrant(reduced, args.collection)


if __name__ == "__main__":
    main()
