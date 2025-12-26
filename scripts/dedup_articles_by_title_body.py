#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path


_WS_RE = re.compile(r"\s+")


def _normalize_text(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    return _WS_RE.sub(" ", text)


def _digest_key(title: str, body: str) -> str:
    normalized = f"{_normalize_text(title)}\n{_normalize_text(body)}".encode("utf-8")
    return hashlib.sha1(normalized).hexdigest()


@dataclass(frozen=True)
class DedupStats:
    kept_rows: int
    dropped_rows: int
    kept_article_ids: int
    dropped_article_ids: int
    unique_title_body: int


def dedup_csv_by_title_body(
    input_path: Path,
    output_path: Path,
    *,
    report_path: Path | None = None,
    id_col: str = "ID",
    title_col: str = "TITLE",
    body_col: str = "BODY",
) -> DedupStats:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)

    # Map from (TITLE,BODY) digest -> kept article ID
    digest_to_kept_id: dict[str, str] = {}
    kept_ids: set[str] = set()
    dropped_ids: set[str] = set()

    kept_rows = 0
    dropped_rows = 0

    with input_path.open("r", encoding="utf-8", newline="") as f_in:
        reader = csv.DictReader(f_in)
        if reader.fieldnames is None:
            raise ValueError("CSV header is missing.")

        missing_cols = [c for c in (id_col, title_col, body_col) if c not in reader.fieldnames]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        with output_path.open("w", encoding="utf-8", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
            writer.writeheader()

            for row in reader:
                article_id = (row.get(id_col) or "").strip()
                title = row.get(title_col) or ""
                body = row.get(body_col) or ""

                digest = _digest_key(title, body)
                kept_id = digest_to_kept_id.get(digest)

                if kept_id is None:
                    digest_to_kept_id[digest] = article_id
                    kept_ids.add(article_id)
                    writer.writerow(row)
                    kept_rows += 1
                    continue

                if kept_id == article_id:
                    writer.writerow(row)
                    kept_rows += 1
                    continue

                dropped_ids.add(article_id)
                dropped_rows += 1

    if report_path is not None:
        with report_path.open("w", encoding="utf-8", newline="") as f_rep:
            w = csv.writer(f_rep)
            w.writerow(["dropped_id"])
            for dropped_id in sorted(dropped_ids):
                w.writerow([dropped_id])

    # Note: kept_ids includes IDs that are first-seen for a digest, but the same ID
    # may appear with multiple digests if upstream data is inconsistent. That's rare,
    # but we keep the stats conservative.
    return DedupStats(
        kept_rows=kept_rows,
        dropped_rows=dropped_rows,
        kept_article_ids=len(kept_ids),
        dropped_article_ids=len(dropped_ids),
        unique_title_body=len(digest_to_kept_id),
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Deduplicate article rows by TITLE+BODY at article-level (ID).")
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--report-dropped-ids", type=Path, default=None)
    p.add_argument("--id-col", default="ID")
    p.add_argument("--title-col", default="TITLE")
    p.add_argument("--body-col", default="BODY")
    args = p.parse_args()

    stats = dedup_csv_by_title_body(
        args.input,
        args.output,
        report_path=args.report_dropped_ids,
        id_col=args.id_col,
        title_col=args.title_col,
        body_col=args.body_col,
    )
    print("done")
    print(f"kept_rows={stats.kept_rows}")
    print(f"dropped_rows={stats.dropped_rows}")
    print(f"unique_title_body={stats.unique_title_body}")
    print(f"kept_article_ids={stats.kept_article_ids}")
    print(f"dropped_article_ids={stats.dropped_article_ids}")


if __name__ == "__main__":
    main()
