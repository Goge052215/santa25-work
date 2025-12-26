import sys
from pathlib import Path

def main():
    project_root = Path(__file__).resolve().parents[1]
    polygons_dir = project_root / "data" / "polygons"
    src = polygons_dir / "0.txt"
    if not src.exists():
        print(f"Source polygon not found: {src}")
        sys.exit(1)
    content = src.read_text()
    created = 0
    for i in range(0, 440):
        dst = polygons_dir / f"{i}.txt"
        if not dst.exists():
            dst.write_text(content)
            created += 1
    print(f"Polygons generated: {created} (cloned from 0.txt) in {polygons_dir}")

if __name__ == "__main__":
    main()
