#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


_WS_RE = re.compile(r"\s+")
_SUSPICIOUS_NAME_RE = re.compile(
    r"(事件|容疑者|被告|会社|組|連合|署|県警|警察|市役所|裁判所|地裁|高裁|最高裁)$"
)

ALLOWED_CATEGORIES_DEFAULT = ["法令違反", "PEPS", "破産", "その他"]


DEFAULT_DIRTY_MARKERS = [
    "ログイン",
    "マイページ",
    "利用規約",
    "サイトマップ",
    "個人情報",
    "著作権",
    "Copyright",
    "アクセスランキング",
    "速報",
    "会員登録",
    "プレゼント",
    "設定",
    "検索",
    "一覧",
]


def _normalize_text(text: str) -> str:
    return _WS_RE.sub(" ", (text or "").strip())


def _is_dirty_body(text: str, markers: list[str], *, min_hits: int) -> bool:
    t = text or ""
    hits = 0
    for m in markers:
        if m in t:
            hits += 1
            if hits >= min_hits:
                return True
    return False


def _domain_from_url(url: str) -> str:
    try:
        return urlparse(url).netloc or ""
    except Exception:
        return ""


@dataclass
class ArticleAgg:
    title: str = ""
    body: str = ""
    article: str = ""
    posted_at: str = ""
    url: str = ""
    domain: str = ""
    dirty_body: bool = False

    rows: int = 0
    offenders: int = 0
    non_offenders: int = 0

    # person-field completeness over all rows (treat each row as a person candidate)
    filled_age: int = 0
    filled_address: int = 0
    filled_affiliation: int = 0
    filled_position: int = 0
    filled_category: int = 0
    filled_act: int = 0

    invalid_category_rows: int = 0
    offender_missing_act: int = 0
    offender_missing_category: int = 0
    suspicious_name_rows: int = 0

    # article-level fields
    occurred_nonempty_rows: int = 0
    police_nonempty_rows: int = 0
    occurred_values: Counter[str] | None = None
    police_values: Counter[str] | None = None
    categories: Counter[str] | None = None

    def __post_init__(self) -> None:
        self.occurred_values = Counter()
        self.police_values = Counter()
        self.categories = Counter()

    def add_row(self, row: dict[str, str], *, allowed_categories: set[str]) -> None:
        self.rows += 1

        if not self.title:
            self.title = row.get("TITLE", "") or ""
        if not self.body:
            self.body = row.get("BODY", "") or ""
        if not self.article:
            self.article = row.get("ARTICLE", "") or ""
        if not self.posted_at:
            self.posted_at = row.get("POSTED_AT", "") or ""
        if not self.url:
            self.url = row.get("URL", "") or ""
            self.domain = _domain_from_url(self.url)

        name = _normalize_text(row.get("P_NAME", "") or "")
        if name and _SUSPICIOUS_NAME_RE.search(name):
            self.suspicious_name_rows += 1

        is_offender = (row.get("IS_OFFENDER", "") or "").strip().lower() == "true"
        if is_offender:
            self.offenders += 1
        else:
            self.non_offenders += 1

        age = (row.get("P_AGE", "") or "").strip()
        if age:
            self.filled_age += 1

        address = (row.get("P_ADDRESS", "") or "").strip()
        if address:
            self.filled_address += 1

        affiliation = (row.get("P_AFFILIATION", "") or "").strip()
        if affiliation:
            self.filled_affiliation += 1

        position = (row.get("P_POSITION", "") or "").strip()
        if position:
            self.filled_position += 1

        category = (row.get("P_CATEGORY", "") or "").strip()
        if category:
            if category in allowed_categories:
                self.filled_category += 1
                self.categories[category] += 1
            else:
                self.invalid_category_rows += 1
        elif is_offender:
            self.offender_missing_category += 1

        act = (row.get("P_DESCRIPTION", "") or "").strip()
        if act:
            self.filled_act += 1
        elif is_offender:
            self.offender_missing_act += 1

        occurred = (row.get("OCCURED_LOCATIONS", "") or "").strip()
        if occurred:
            self.occurred_nonempty_rows += 1
            for part in occurred.split(","):
                v = part.strip()
                if v:
                    self.occurred_values[v] += 1

        police = (row.get("POLICE_DEPARTMENTS", "") or "").strip()
        if police:
            self.police_nonempty_rows += 1
            for part in police.split(","):
                v = part.strip()
                if v:
                    self.police_values[v] += 1

    def finalize(self, dirty_markers: list[str], *, dirty_min_hits: int) -> None:
        if self.body and not self.dirty_body:
            self.dirty_body = _is_dirty_body(self.body, dirty_markers, min_hits=dirty_min_hits)

    def primary_category(self) -> str:
        if not self.categories:
            return ""
        if not self.categories:
            return ""
        return self.categories.most_common(1)[0][0] if self.categories else ""

    def completeness_score(self) -> float:
        # Score focuses on how "filled" a training sample would be.
        if self.rows <= 0:
            return 0.0

        # Per-row fill rates
        r = float(self.rows)
        score = 0.0
        score += 1.5 * (self.filled_act / r)
        score += 1.2 * (self.filled_category / r)
        score += 0.9 * (self.filled_address / r)
        score += 0.6 * (self.filled_position / r)
        score += 0.6 * (self.filled_affiliation / r)
        score += 0.4 * (self.filled_age / r)

        # Article-level bonuses (rare, useful if present)
        score += 0.4 if (self.occurred_nonempty_rows > 0) else 0.0
        score += 0.4 if (self.police_nonempty_rows > 0) else 0.0

        # Penalize obviously problematic patterns
        score -= 2.0 * (self.offender_missing_act / r)
        score -= 1.0 * (self.offender_missing_category / r)
        score -= 0.6 * (self.invalid_category_rows / r)
        score -= 0.3 * (self.suspicious_name_rows / r)

        # Prefer having at least one offender (training signal), but keep some non-offender-only too.
        score += 0.3 if self.offenders > 0 else -0.1

        # Body quality
        if self.dirty_body:
            score -= 1.5
        return score

    def risk_score(self) -> float:
        if self.rows <= 0:
            return 0.0
        r = float(self.rows)
        score = 0.0
        score += 2.0 * (self.offender_missing_act / r)
        score += 1.0 * (self.offender_missing_category / r)
        score += 1.0 * (self.invalid_category_rows / r)
        score += 0.5 * (self.suspicious_name_rows / r)
        score += 1.0 if self.dirty_body else 0.0
        return score


def _stratified_pick(
    rng: random.Random,
    buckets: dict[str, list[str]],
    *,
    target_n: int,
    already: set[str],
) -> list[str]:
    selected: list[str] = []
    bucket_keys = [k for k, v in buckets.items() if v]
    rng.shuffle(bucket_keys)

    # Round-robin pick
    while len(selected) < target_n and bucket_keys:
        progressed = False
        for k in list(bucket_keys):
            if len(selected) >= target_n:
                break
            lst = buckets[k]
            while lst and lst[-1] in already:
                lst.pop()
            if not lst:
                bucket_keys.remove(k)
                continue
            picked = lst.pop()
            if picked in already:
                continue
            selected.append(picked)
            already.add(picked)
            progressed = True
        if not progressed:
            break
    return selected


def main() -> None:
    p = argparse.ArgumentParser(description="Select ~N article IDs for human review from a person-level CSV.")
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output-ids", required=True, type=Path)
    p.add_argument("--output-rows", required=True, type=Path)
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--complete-n", type=int, default=1500)
    p.add_argument("--coverage-n", type=int, default=300)
    p.add_argument("--risk-n", type=int, default=100)
    p.add_argument("--random-n", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--exclude-dirty", action="store_true", default=False)
    p.add_argument("--dirty-markers", type=str, default="|".join(DEFAULT_DIRTY_MARKERS))
    p.add_argument("--dirty-min-hits", type=int, default=3)
    p.add_argument("--allowed-categories", type=str, default="|".join(ALLOWED_CATEGORIES_DEFAULT))
    args = p.parse_args()

    if args.complete_n + args.coverage_n + args.risk_n + args.random_n != args.n:
        raise SystemExit("complete-n + coverage-n + risk-n + random-n must equal n")

    dirty_markers = [m for m in args.dirty_markers.split("|") if m]
    allowed_categories = {c for c in args.allowed_categories.split("|") if c}

    aggs: dict[str, ArticleAgg] = {}

    with args.input.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            article_id = (row.get("ID") or "").strip()
            if not article_id:
                continue
            agg = aggs.get(article_id)
            if agg is None:
                agg = ArticleAgg()
                aggs[article_id] = agg
            agg.add_row(row, allowed_categories=allowed_categories)

    for agg in aggs.values():
        agg.finalize(dirty_markers, dirty_min_hits=args.dirty_min_hits)

    # Prepare ranked lists
    all_ids = list(aggs.keys())
    if args.exclude_dirty:
        all_ids = [aid for aid in all_ids if not aggs[aid].dirty_body]
    rng = random.Random(args.seed)

    def complete_sort_key(aid: str) -> tuple[float, int, int]:
        agg = aggs[aid]
        return (agg.completeness_score(), agg.offenders, agg.rows)

    complete_ids = sorted(all_ids, key=complete_sort_key, reverse=True)

    def risk_sort_key(aid: str) -> tuple[float, int, int]:
        agg = aggs[aid]
        return (agg.risk_score(), agg.offenders, agg.rows)

    risk_ids = sorted(all_ids, key=risk_sort_key, reverse=True)

    selected: list[str] = []
    selected_set: set[str] = set()

    # 1) completeness-heavy
    for aid in complete_ids:
        if len(selected) >= args.complete_n:
            break
        if aid in selected_set:
            continue
        selected.append(aid)
        selected_set.add(aid)

    # 2) coverage: stratify remaining by (category, offender_count_bin, domain)
    def offender_bin(offenders: int) -> str:
        if offenders <= 0:
            return "off0"
        if offenders == 1:
            return "off1"
        if offenders == 2:
            return "off2"
        return "off3p"

    buckets: dict[str, list[str]] = defaultdict(list)
    remaining = [aid for aid in all_ids if aid not in selected_set]
    rng.shuffle(remaining)
    for aid in remaining:
        agg = aggs[aid]
        cat = agg.primary_category() or "cat_empty"
        ob = offender_bin(agg.offenders)
        dom = agg.domain or "domain_empty"
        key = f"{cat}|{ob}|{dom}"
        buckets[key].append(aid)
    # shuffle within each bucket so pop() is random-ish but deterministic by seed
    for k in list(buckets.keys()):
        rng.shuffle(buckets[k])

    selected.extend(_stratified_pick(rng, buckets, target_n=args.coverage_n, already=selected_set))

    # 3) small risk set (keep low to avoid poisoning, but good for finding systematic fixes)
    for aid in risk_ids:
        if len(selected) >= args.complete_n + args.coverage_n + args.risk_n:
            break
        if aid in selected_set:
            continue
        selected.append(aid)
        selected_set.add(aid)

    # 4) pure random from remaining
    remaining2 = [aid for aid in all_ids if aid not in selected_set]
    rng.shuffle(remaining2)
    for aid in remaining2[: args.random_n]:
        selected.append(aid)
        selected_set.add(aid)

    selected = selected[: args.n]

    # Write selected IDs with some diagnostics
    args.output_ids.parent.mkdir(parents=True, exist_ok=True)
    with args.output_ids.open("w", encoding="utf-8", newline="") as f_out:
        w = csv.writer(f_out)
        w.writerow(
            [
                "ID",
                "completeness_score",
                "risk_score",
                "rows",
                "offenders",
                "primary_category",
                "domain",
                "dirty_body",
            ]
        )
        for aid in selected:
            agg = aggs[aid]
            w.writerow(
                [
                    aid,
                    f"{agg.completeness_score():.4f}",
                    f"{agg.risk_score():.4f}",
                    agg.rows,
                    agg.offenders,
                    agg.primary_category(),
                    agg.domain,
                    "1" if agg.dirty_body else "0",
                ]
            )

    # Write filtered rows for those IDs (same columns as input)
    wanted = set(selected)
    args.output_rows.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open("r", encoding="utf-8", newline="") as f_in:
        reader = csv.DictReader(f_in)
        if reader.fieldnames is None:
            raise SystemExit("missing header")
        with args.output_rows.open("w", encoding="utf-8", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                if (row.get("ID") or "").strip() in wanted:
                    writer.writerow(row)

    print("done")
    print(f"total_articles={len(all_ids)}")
    print(f"selected_articles={len(selected)}")
    print(f"outputs: {args.output_ids} , {args.output_rows}")


if __name__ == "__main__":
    main()
