"""
Download the current set of 300 portrait photos from the big-web-simulation
(3 sites x 100 people, seeded by site label) into templates/photos/.
"""
import os
import urllib.request
import json
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "templates" / "photos"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SITES = [
    ("site1", 100),
    ("site2", 100),
    ("site3", 100),
]

def fetch_users(seed, count):
    url = (
        f"https://randomuser.me/api/"
        f"?results={count}&seed={seed}&inc=gender,name,picture"
    )
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read())["results"]

def download_image(img_url, dest_path):
    with urllib.request.urlopen(img_url, timeout=30) as resp:
        dest_path.write_bytes(resp.read())

total = 0
for seed, count in SITES:
    print(f"Fetching {count} users for seed='{seed}' ...")
    users = fetch_users(seed, count)
    for i, user in enumerate(users):
        img_url = user.get("picture", {}).get("large", "")
        if not img_url:
            print(f"  [skip] {seed} #{i+1}: no image URL")
            continue

        first = user.get("name", {}).get("first", "unknown")
        last  = user.get("name", {}).get("last",  "unknown")
        ext   = img_url.rsplit(".", 1)[-1].split("?")[0] or "jpg"
        filename = f"{seed}_{i+1:03d}_{first}_{last}.{ext}"
        dest = OUTPUT_DIR / filename

        try:
            download_image(img_url, dest)
            print(f"  [{i+1}/{count}] saved {filename}")
            total += 1
        except Exception as e:
            print(f"  [error] {seed} #{i+1}: {e}")

print(f"\nDone. {total} photos saved to {OUTPUT_DIR}")
