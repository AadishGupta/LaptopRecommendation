"""
rebuild_pkl.py  —  run once to convert LangChain pkl to flat list format
"""
import pickle, faiss, numpy as np, pandas as pd, re, os

FAISS_PATH = "data/index.faiss"
PKL_PATH   = "data/index.pkl"
CSV_PATH   = "laptop_data.csv"

CHUNK_TOPICS = {
    "GPU":          ["gpu","graphics","rtx","gtx","radeon","vram","nvidia","amd gpu","intel arc","uhd graphics"],
    "Display":      ["display","screen","resolution","fhd","4k","retina","oled","ips","nits","hz","refresh"],
    "Portability":  ["weight","kg","portable","thin","light","slim","compact","battery"],
    "Multitasking": ["ram","gb ram","memory","multitask"],
    "Processing":   ["cpu","processor","core i","ryzen","intel","amd","ghz","m1","m2","m3"],
}

def context_based_chunking(description):
    sentences = re.split(r"[.,;]\s*", description)
    sentences = [s.strip() for s in sentences if s.strip()]
    buckets = {t: [] for t in CHUNK_TOPICS}
    misc = []
    for sentence in sentences:
        lower = sentence.lower()
        matched = False
        for topic, keywords in CHUNK_TOPICS.items():
            if any(kw in lower for kw in keywords):
                buckets[topic].append(sentence)
                matched = True
                break
        if not matched:
            misc.append(sentence)
    chunks = []
    for topic, sents in buckets.items():
        if sents:
            chunks.append(f"{topic} info: " + ". ".join(sents))
    if misc:
        chunks.append("General info: " + ". ".join(misc))
    chunks.append(description)
    return chunks

# ── Step 1: load pkl and inspect its actual structure ──────────────────────
print("Loading pkl...")
with open(PKL_PATH, "rb") as f:
    lc_data = pickle.load(f)

print(f"  Type of pkl content : {type(lc_data)}")
if isinstance(lc_data, tuple):
    print(f"  Tuple length        : {len(lc_data)}")
    for i, item in enumerate(lc_data):
        print(f"  lc_data[{i}] type    : {type(item)}  len={len(item) if hasattr(item, '__len__') else 'n/a'}")

# ── Determine docstore dict and index_to_doc_id ────────────────────────────
if isinstance(lc_data, tuple) and len(lc_data) == 2:
    part0, part1 = lc_data

    # part0 might be the raw dict or a docstore object
    if isinstance(part0, dict):
        docstore_dict   = part0           # {doc_id: Document}
        index_to_doc_id = part1           # {faiss_int: doc_id}
    elif hasattr(part0, "_dict"):
        docstore_dict   = part0._dict
        index_to_doc_id = part1
    elif hasattr(part0, "docstore"):
        docstore_dict   = part0.docstore._dict
        index_to_doc_id = part1
    else:
        raise RuntimeError(f"Cannot interpret lc_data[0] of type {type(part0)}")
else:
    raise RuntimeError(f"Unexpected pkl structure: {type(lc_data)}")

print(f"  docstore_dict entries : {len(docstore_dict)}")
print(f"  index_to_doc_id size  : {len(index_to_doc_id)}")

# Peek at first document
sample_id  = next(iter(docstore_dict))
sample_doc = docstore_dict[sample_id]
print(f"\n  Sample doc type : {type(sample_doc)}")
if hasattr(sample_doc, "page_content"):
    print(f"  page_content    : {sample_doc.page_content[:120]!r}")
    print(f"  metadata        : {sample_doc.metadata}")
else:
    print(f"  raw value       : {str(sample_doc)[:200]!r}")

# ── Step 2: load CSV ───────────────────────────────────────────────────────
print(f"\nLoading CSV '{CSV_PATH}'...")
df = pd.read_csv(CSV_PATH)
df["Price"] = df["Price"].astype(str).str.replace(",", "").astype(int)
print(f"  Columns : {list(df.columns)}")
print(f"  Rows    : {len(df)}")

desc_to_row = {str(r["Description"]).strip(): r for _, r in df.iterrows()}
name_to_row = {str(r["Name"]).strip(): r for _, r in df.iterrows()} if "Name" in df.columns else {}

# ── Step 3: load FAISS index ───────────────────────────────────────────────
print(f"\nLoading FAISS index '{FAISS_PATH}'...")
index = faiss.read_index(FAISS_PATH)
dim   = index.d
print(f"  {index.ntotal} vectors, dim={dim}")

# ── Step 4: build flat list ────────────────────────────────────────────────
print("\nBuilding records...")
vector_store = []
matched_desc = matched_meta = unmatched = 0

for faiss_idx, doc_id in index_to_doc_id.items():
    doc = docstore_dict[doc_id]

    if hasattr(doc, "page_content"):
        description = doc.page_content.strip()
        metadata    = doc.metadata
    else:
        description = str(doc).strip()
        metadata    = {}

    vec = np.zeros((1, dim), dtype="float32")
    index.reconstruct(int(faiss_idx), vec[0])
    full_embedding = vec[0].tolist()

    csv_row = desc_to_row.get(description)
    if csv_row is not None:
        matched_desc += 1
        name  = str(csv_row.get("Name",  metadata.get("name",  f"Laptop_{faiss_idx}")))
        price = int(csv_row.get("Price", metadata.get("price", 0)))
    else:
        meta_name = str(metadata.get("name", metadata.get("Name", ""))).strip()
        csv_row   = name_to_row.get(meta_name) if meta_name else None
        if csv_row is not None:
            matched_meta += 1
            name  = meta_name
            price = int(csv_row.get("Price", metadata.get("price", 0)))
        else:
            unmatched += 1
            name  = str(metadata.get("name", metadata.get("Name", f"Laptop_{faiss_idx}")))
            price = int(metadata.get("price", metadata.get("Price", 0)))

    chunks = context_based_chunking(description)
    vector_store.append({
        "id":               int(faiss_idx),
        "name":             name,
        "description":      description,
        "price":            price,
        "full_embedding":   full_embedding,
        "chunks":           chunks,
        "chunk_embeddings": [full_embedding] * len(chunks),
    })

print(f"  Built         : {len(vector_store)} records")
print(f"  Matched desc  : {matched_desc}")
print(f"  Matched meta  : {matched_meta}")
print(f"  Unmatched     : {unmatched}")

print("\nSample records:")
for rec in vector_store[:3]:
    print(f"  id={rec['id']}  price={rec['price']}  name={rec['name']!r}")
    print(f"  desc={rec['description'][:100]!r}")

# ── Step 5: save ───────────────────────────────────────────────────────────
backup = PKL_PATH + ".langchain_backup"
if not os.path.exists(backup):
    os.rename(PKL_PATH, backup)
    print(f"\nBacked up original pkl to '{backup}'")
else:
    os.remove(PKL_PATH)

with open(PKL_PATH, "wb") as f:
    pickle.dump(vector_store, f)

print(f"Done. '{PKL_PATH}' now has {len(vector_store)} laptops.")