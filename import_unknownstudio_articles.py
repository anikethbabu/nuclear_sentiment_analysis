from __future__ import annotations

import argparse
import hashlib
import sqlite3
from pathlib import Path


ARTICLE_FOLDERS = {
    "ANS": Path("ans_articles"),
    "World Nuclear": Path("World_Nuclear_Scraper") / "articles",
}


def normalize_text(text: str) -> str:
    return " ".join(text.replace("\ufeff", "").split()).strip()


def article_id(source: str, filename: str) -> str:
    return f"{source}::{filename}"


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def import_articles(db_path: Path, external_root: Path) -> tuple[int, int]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            article_id TEXT UNIQUE NOT NULL,
            source TEXT NOT NULL,
            filename TEXT NOT NULL,
            title TEXT NOT NULL,
            path TEXT NOT NULL,
            content TEXT NOT NULL,
            content_sha256 TEXT NOT NULL,
            word_count INTEGER NOT NULL
        )
        """
    )

    seen_hashes: set[str] = set()
    inserted = 0
    skipped_duplicates = 0

    for source, relative_folder in ARTICLE_FOLDERS.items():
        folder = external_root / relative_folder
        if not folder.is_dir():
            continue
        for file_path in sorted(folder.glob("*.txt")):
            text = normalize_text(file_path.read_text(encoding="utf-8", errors="replace"))
            if not text:
                continue
            digest = content_hash(text)
            if digest in seen_hashes:
                skipped_duplicates += 1
            seen_hashes.add(digest)
            title = file_path.stem.replace("-", " ").strip()
            cur.execute(
                """
                INSERT OR REPLACE INTO articles
                (article_id, source, filename, title, path, content, content_sha256, word_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    article_id(source, file_path.name),
                    source,
                    file_path.name,
                    title,
                    str(file_path),
                    text,
                    digest,
                    len(text.split()),
                ),
            )
            inserted += 1

    con.commit()
    con.close()
    return inserted, skipped_duplicates


def main() -> None:
    parser = argparse.ArgumentParser(description="Import repository article text files into SQLite.")
    parser.add_argument("--external-root", default=str(Path("..") / "UnknowStudio4"))
    parser.add_argument("--db", default="models/unknownstudio/unknownstudio_articles.db")
    args = parser.parse_args()

    inserted, skipped_duplicates = import_articles(Path(args.db), Path(args.external_root))
    print(f"imported_or_updated={inserted}")
    print(f"duplicate_content_seen={skipped_duplicates}")
    print(f"database={args.db}")


if __name__ == "__main__":
    main()
