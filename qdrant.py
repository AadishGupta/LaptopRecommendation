from qdrant_client import QdrantClient

try:
    client = QdrantClient(host='localhost', port=6333)
    print("✅ Connected to Qdrant")
    
    info = client.get_collection('laptops_chunked')
    
    # Try different attribute names
    if hasattr(info, 'points_count'):
        print(f"✅ Vectors: {info.points_count:,}")
    elif hasattr(info, 'vectors_count'):
        print(f"✅ Vectors: {info.vectors_count:,}")
    else:
        print(f"✅ Collection info: {info}")
    
except Exception as e:
    print(f"❌ Error: {e}")