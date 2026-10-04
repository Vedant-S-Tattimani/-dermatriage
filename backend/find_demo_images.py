import csv
from pathlib import Path

rows = []
with open('reports/confidence_benchmark.csv') as f:
    reader = csv.DictReader(f)
    for r in reader:
        r['final_confidence'] = float(r['final_confidence'])
        r['raw_confidence'] = float(r['raw_confidence'])
        rows.append(r)

# Sort by final confidence descending
rows.sort(key=lambda x: x['final_confidence'], reverse=True)

# Get best 3 per class
seen_classes = {}
best = []
for r in rows:
    c = r['predicted_class']
    if c not in seen_classes:
        seen_classes[c] = 0
    if seen_classes[c] < 3:
        best.append(r)
        seen_classes[c] += 1

print("=" * 85)
print("BEST IMAGES PER CLASS (top 3 each)")
print("=" * 85)
current_class = None
for r in sorted(best, key=lambda x: (x['predicted_class'], -x['final_confidence'])):
    if r['predicted_class'] != current_class:
        current_class = r['predicted_class']
        print(f"\n  [{current_class.upper()}]")
    conf = r['final_confidence'] * 100
    img = r['image']
    path = Path('uploads') / img
    print(f"    {img:<45}  {conf:>5.1f}%   ->  {path}")

print("\n")
print("=" * 85)
print("TOP 10 OVERALL — USE THESE FOR YOUR DEMO")
print("=" * 85)
for i, r in enumerate(rows[:10], 1):
    conf = r['final_confidence'] * 100
    img = r['image']
    cls = r['predicted_class']
    print(f"  {i:>2}. {img:<45}  {cls:<10}  {conf:>5.1f}%")

print("\n")
print("=" * 85)
print("FILE PATHS (copy-paste ready)")
print("=" * 85)
for r in rows[:10]:
    print(f"  d:\\medical assiastant\\backend\\uploads\\{r['image']}")
