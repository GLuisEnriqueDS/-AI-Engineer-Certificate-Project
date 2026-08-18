import argparse
import subprocess
import sys
from pathlib import Path

INGEST_DIR = Path(__file__).resolve().parent


def run_step(script_name, extra_args=None):
    cmd = [sys.executable, str(INGEST_DIR / script_name), *(extra_args or [])]
    print(f"\n=== {' '.join(cmd)} ===")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Corre el pipeline de ingesta completo: fetch_articles -> fetch_content -> index_to_chroma.")
    parser.add_argument("--top-n", type=int, help="Se pasa a fetch_content.py: cuantos articulos recientes procesar.")
    parser.add_argument("--force", action="store_true", help="Se pasa a fetch_content.py e index_to_chroma.py: reprocesa todo, incluso lo ya hecho.")
    args = parser.parse_args()

    fetch_content_args = []
    if args.top_n is not None:
        fetch_content_args += ["--top-n", str(args.top_n)]
    if args.force:
        fetch_content_args.append("--force")

    index_args = ["--force"] if args.force else []

    run_step("fetch_articles.py")
    run_step("fetch_content.py", fetch_content_args)
    run_step("index_to_chroma.py", index_args)

    print("\nPipeline completo.")


if __name__ == "__main__":
    main()
