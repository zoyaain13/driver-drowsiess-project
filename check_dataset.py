from pathlib import Path

dataset = Path.home() / "Downloads" / "archive"

print("Dataset location:", dataset)
print("Dataset exists:", dataset.exists())

for folder in ["train", "val", "test"]:
    path = dataset / folder
    print(f"\n{folder}:")
    print("  exists:", path.exists())

    for category in ["open", "closed"]:
        category_path = path / category
        if category_path.exists():
            images = list(category_path.glob("*.png"))
            print(f"  {category}: {len(images)} images")
        else:
            print(f"  {category}: NOT FOUND")