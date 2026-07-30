"""
filter_pkl.py  —  keep only laptop records, drop everything else
"""
import pickle, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PKL_PATH = "data/index.pkl"

with open(PKL_PATH, "rb") as f:
    records = pickle.load(f)

print(f"Total records before filter: {len(records)}")

# Count categories
from collections import Counter
cats = Counter()
for r in records:
    first_line = r["description"].split("\n")[0].strip()
    cats[first_line] += 1
print("Categories found:")
for cat, count in cats.most_common():
    print(f"  {cat!r}: {count}")

# Keep only laptops with a real price
laptops = [r for r in records
           if r["description"].startswith("CATEGORY: laptops") and r["price"] > 0]

print(f"\nLaptops with valid price: {len(laptops)}")
print("\nSample:")
for r in laptops[:3]:
    print(f"  id={r['id']}  price={r['price']}  name={r['name']!r}")

# Re-index ids sequentially
for i, r in enumerate(laptops):
    r["id"] = i

import shutil, os
backup = PKL_PATH + ".mixed_backup"
if not os.path.exists(backup):
    shutil.copy(PKL_PATH, backup)
    print(f"\nBacked up to '{backup}'")

with open(PKL_PATH, "wb") as f:
    pickle.dump(laptops, f)

print(f"Done. '{PKL_PATH}' now has {len(laptops)} laptops only.")
