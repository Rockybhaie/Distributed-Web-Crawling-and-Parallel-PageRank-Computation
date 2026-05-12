import json, pickle

# Check JSON
with open("output/web_graph.json") as f:
    adj = json.load(f)

print(f"Pages crawled: {len(adj)}")
print(f"Total edges:   {sum(len(v) for v in adj.values())}")

# Print 3 sample pages and their links
for i, (page, links) in enumerate(adj.items()):
    print(f"\n{page}\n  → {links[:3]}")
    if i == 2:
        break