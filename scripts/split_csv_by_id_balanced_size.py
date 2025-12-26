#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import os
from collections import defaultdict
from pathlib import Path


def _estimate_serialized_bytes(rows: list[dict[str, str]], fieldnames: list[str]) -> int:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fieldnames, lineterminator="\n")
    for row in rows:
        w.writerow(row)
    return len(buf.getvalue().encode("utf-8"))


def main() -> None:
    p = argparse.ArgumentParser(
        description="Split a CSV into 2 parts by ID, balancing output size (no ID is split across files)."
    )
    p.add_argument("--input-rows", required=True, type=Path)
    p.add_argument("--input-ids", required=True, type=Path)
    p.add_argument("--output-rows-a", required=True, type=Path)
    p.add_argument("--output-rows-b", required=True, type=Path)
    p.add_argument("--output-ids-a", required=True, type=Path)
    p.add_argument("--output-ids-b", required=True, type=Path)
    p.add_argument("--id-col", default="ID")
    args = p.parse_args()

    # Load ID metadata (keeps original order)
    with args.input_ids.open("r", encoding="utf-8", newline="") as f:
        ids_reader = csv.DictReader(f)
        ids_header = ids_reader.fieldnames
        if not ids_header:
            raise SystemExit("missing header in input-ids")
        id_rows: list[dict[str, str]] = []
        id_order: list[str] = []
        for row in ids_reader:
            aid = (row.get(args.id_col) or "").strip()
            if not aid:
                continue
            id_rows.append(row)
            id_order.append(aid)

    # Load rows grouped by ID
    with args.input_rows.open("r", encoding="utf-8", newline="") as f:
        rows_reader = csv.DictReader(f)
        rows_header = rows_reader.fieldnames
        if not rows_header:
            raise SystemExit("missing header in input-rows")
        if args.id_col not in rows_header:
            raise SystemExit(f"missing id column in input-rows: {args.id_col}")

        rows_by_id: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows_reader:
            aid = (row.get(args.id_col) or "").strip()
            if not aid:
                continue
            rows_by_id[aid].append(row)

    missing = [aid for aid in id_order if aid not in rows_by_id]
    if missing:
        raise SystemExit(f"{len(missing)} IDs from input-ids are missing in input-rows (example: {missing[0]})")

    # Estimate per-ID byte contribution and do greedy bin packing
    sizes: dict[str, int] = {}
    for aid in id_order:
        sizes[aid] = _estimate_serialized_bytes(rows_by_id[aid], rows_header)

    ids_sorted = sorted(id_order, key=lambda x: sizes[x], reverse=True)
    a_ids: set[str] = set()
    b_ids: set[str] = set()
    a_bytes = 0
    b_bytes = 0
    for aid in ids_sorted:
        if a_bytes <= b_bytes:
            a_ids.add(aid)
            a_bytes += sizes[aid]
        else:
            b_ids.add(aid)
            b_bytes += sizes[aid]

    # Write outputs in the original ID order for readability
    for out_path in (
        args.output_rows_a,
        args.output_rows_b,
        args.output_ids_a,
        args.output_ids_b,
    ):
        out_path.parent.mkdir(parents=True, exist_ok=True)

    with args.output_ids_a.open("w", encoding="utf-8", newline="") as fa, args.output_ids_b.open(
        "w", encoding="utf-8", newline=""
    ) as fb:
        wa = csv.DictWriter(fa, fieldnames=ids_header, lineterminator="\n")
        wb = csv.DictWriter(fb, fieldnames=ids_header, lineterminator="\n")
        wa.writeheader()
        wb.writeheader()
        for row in id_rows:
            aid = (row.get(args.id_col) or "").strip()
            if aid in a_ids:
                wa.writerow(row)
            else:
                wb.writerow(row)

    with args.output_rows_a.open("w", encoding="utf-8", newline="") as fa, args.output_rows_b.open(
        "w", encoding="utf-8", newline=""
    ) as fb:
        wa = csv.DictWriter(fa, fieldnames=rows_header, lineterminator="\n")
        wb = csv.DictWriter(fb, fieldnames=rows_header, lineterminator="\n")
        wa.writeheader()
        wb.writeheader()
        for aid in id_order:
            if aid in a_ids:
                for row in rows_by_id[aid]:
                    wa.writerow(row)
            else:
                for row in rows_by_id[aid]:
                    wb.writerow(row)

    size_a = os.path.getsize(args.output_rows_a)
    size_b = os.path.getsize(args.output_rows_b)
    print("done")
    print(f"part_a_ids={len(a_ids)} part_b_ids={len(b_ids)}")
    print(f"part_a_bytes={size_a} part_b_bytes={size_b}")


if __name__ == "__main__":
    main()

