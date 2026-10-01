#!/usr/bin/env python3
"""Build PRISM's broad initial quarterly feature universe.

Philosophy
----------
The ICI target is deliberately strict and comparable. Predictors are deliberately
broad: collect economically meaningful, reproducible signals first, then let later
train-only selection / regularization / ablation decide what survives.

This stage DOES NOT perform final feature selection. It collects and audits:
  * node-level QCEW employment history (lagged target ingredient only),
  * node-level CES labor state (exact where published; structural masks otherwise),
  * node-level industry PPI (exact where published; structural masks otherwise),
  * node-level JOLTS labor-demand/turnover proxies mapped by BLS broad industry,
  * national BLS price/labor/activity context,
  * NY Fed effective federal funds rate,
  * liquid market / FX / commodity / sector-price context from Yahoo Finance.

Every source is aligned by its official publication date to the quarter-end forecast
origin. Fixed 1Q/2Q availability lags are not used. Partial node coverage is retained
with explicit masks instead of being discarded solely because a statistic is not
published for all 56 industries.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from urllib.parse import urljoin


ROOT = Path(__file__).resolve().parent.parent
JOB = "02_initial_feature_pool"
DATA_DIR = ROOT / "Data" / JOB
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT / "Results" / JOB

NODE_MAP = ROOT / "Results" / "01_target_data_collection" / "node_source_mapping.csv"
TARGET_BEA_LONG = ROOT / "Data" / "01_target_data_collection" / "processed" / "bea_quarterly_56_long.csv"
TARGET_SUMMARY = ROOT / "Results" / "01_target_data_collection" / "collection_summary.json"
TARGET_EMPLOYMENT = ROOT / "Data" / "01_target_data_collection" / "processed" / "qcew_quarterly_employment_56.csv"
TARGET_RELEASE_METADATA = ROOT / "Data" / "01_target_data_collection" / "processed" / "target_release_metadata.csv"

BLS_ROOT = "https://download.bls.gov/pub/time.series"
NYFED_EFFR = "https://markets.newyorkfed.org/api/rates/unsecured/effr/search.json"

# Relaxed initial-pool gates: later train-only selection is intentionally stricter.
MIN_TIME_COVERAGE = 0.70
MIN_NODE_FRACTION = 0.40

BLS_HEADERS = {"User-Agent": "PRISM academic research haseung.ryu.ai@gmail.com"}
FED_BUSINESS_DAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())
FEATURE_RELEASE_METADATA = PROC_DIR / "feature_release_metadata.csv"

MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


def align_blocks(
    blocks: dict[str, pd.DataFrame | pd.Series],
    quarters: list[str],
    release_map: dict[str, pd.Timestamp],
) -> dict[str, pd.DataFrame | pd.Series]:
    return {
        name: align_quarterly_panel_by_release(obj, quarters, release_map)
        for name, obj in blocks.items()
    }


def quarter_end(quarter: str) -> pd.Timestamp:
    return pd.Period(quarter, freq="Q").end_time.normalize()


def parse_release_date(text: str, default_year: int) -> pd.Timestamp | None:
    cleaned = re.sub(r"^[A-Za-z]+,\s*", "", str(text).strip())
    if re.search(r"\b\d{4}\b", cleaned):
        try:
            return pd.Timestamp(pd.to_datetime(cleaned, errors="raise")).normalize()
        except Exception:
            pass
    m = re.search(
        r"(Jan\.?|Feb\.?|Mar\.?|March|Apr\.?|May|Jun\.?|June|Jul\.?|July|Aug\.?|"
        r"Sep\.?|Sept\.?|September|Oct\.?|Nov\.?|Dec\.?)\s+(\d{1,2})(?:,\s*(\d{4}))?",
        cleaned,
        re.I,
    )
    if not m:
        return None
    year = int(m.group(3)) if m.group(3) else default_year
    try:
        return pd.Timestamp(pd.to_datetime(f"{m.group(1)} {m.group(2)}, {year}")).normalize()
    except Exception:
        return None


def release_date_from_archive_href(href: str) -> pd.Timestamp | None:
    m = re.search(r"_(\d{2})(\d{2})(\d{4})\.(?:txt|pdf|htm|html)$", href, re.I)
    if not m:
        return None
    try:
        return pd.Timestamp(year=int(m.group(3)), month=int(m.group(1)), day=int(m.group(2)))
    except ValueError:
        return None


def parse_pre_schedule_entry(line: str, default_year: int) -> tuple[str, pd.Timestamp] | None:
    m = re.search(
        r"^(.*?)\s{2,}"
        r"((?:Jan(?:uary)?\.?|Feb(?:ruary)?\.?|Mar(?:ch)?\.?|Apr(?:il)?\.?|May|"
        r"Jun(?:e)?\.?|Jul(?:y)?\.?|Aug(?:ust)?\.?|Sep(?:t(?:ember)?)?\.?|"
        r"Oct(?:ober)?\.?|Nov(?:ember)?\.?|Dec(?:ember)?\.?)\s+\d{1,2}(?:,\s*\d{4})?)"
        r"(?:\s{2,}|\t+)",
        line.strip(),
        re.I,
    )
    if not m:
        return None
    release_date = parse_release_date(m.group(2), default_year)
    if release_date is None:
        return None
    return m.group(1).strip(), release_date


def classify_bls_release(desc: str) -> str | None:
    d = desc.casefold()
    if "employment situation" in d:
        return "EMPLOYMENT_SITUATION"
    if "consumer price index" in d:
        return "CPI"
    if "producer price index" in d:
        return "PPI"
    if "job openings and labor turnover" in d and "state job" not in d:
        return "JOLTS"
    return None


def month_ref(text: str) -> str | None:
    m = re.search(
        r"(?:for\s+|,\s*)(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})",
        text,
        re.I,
    )
    if not m:
        return None
    return f"{int(m.group(2)):04d}-{MONTHS[m.group(1).lower()]:02d}"


def build_bls_monthly_calendar(session: requests.Session, refresh: bool) -> pd.DataFrame:
    cache = RAW_DIR / "BLS" / "feature_release_metadata_bls.csv"
    if cache.exists() and not refresh:
        cached = pd.read_csv(cache, parse_dates=["release_date"])
        required_months = set(pd.period_range("2005-01", "2026-01", freq="M").astype(str))
        complete = True
        for family in ("EMPLOYMENT_SITUATION", "CPI", "PPI", "JOLTS"):
            have = set(cached.loc[cached["family"] == family, "reference_month"].astype(str))
            if not required_months.issubset(have):
                complete = False
                break
        if complete:
            return cached

    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for year in range(2005, 2027):
        url = f"https://www.bls.gov/schedule/{year}/home.htm"
        soup = BeautifulSoup(session.get(url, timeout=30).text, "html.parser")
        entries: list[tuple[str, str]] = []
        for table in soup.find_all("table"):
            for tr in table.find_all("tr"):
                cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
                if len(cells) >= 3 and cells[0].casefold() != "date":
                    entries.append((cells[0], cells[2]))
        pre = soup.find("pre")
        if pre:
            for line in pre.get_text().splitlines():
                family = classify_bls_release(line)
                if family is None:
                    continue
                parsed = parse_pre_schedule_entry(line, year)
                if parsed is not None:
                    desc, release_date = parsed
                    entries.append((release_date.date().isoformat(), desc))
        for date_text, desc in entries:
            family = classify_bls_release(desc)
            ref = month_ref(desc)
            release_date = parse_release_date(date_text, year)
            if ref is not None and release_date is not None and release_date < pd.Timestamp(f"{ref}-01"):
                release_date = release_date + pd.DateOffset(years=1)
            if family is None or ref is None or release_date is None:
                continue
            key = (family, ref)
            if key in seen:
                continue
            rows.append({
                "family": family,
                "reference_month": ref,
                "release_date": release_date,
                "source_url": url,
            })
            seen.add(key)

    # Fill any historical schedule gaps from the official BLS news-release
    # archives. Only missing reference months are fetched.
    archive_specs = {
        "EMPLOYMENT_SITUATION": (
            "empsit",
            r"EMPLOYMENT SITUATION\s*:\s*(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})",
        ),
        "CPI": (
            "cpi",
            r"CONSUMER PRICE INDEX(?:ES)?\s*[:\-]+\s*(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})",
        ),
        "PPI": (
            "ppi",
            r"PRODUCER PRICE INDEX(?:ES)?\s*[:\-]+\s*(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})",
        ),
    }
    required_months = set(pd.period_range("2005-01", "2026-01", freq="M").astype(str))
    for family, (slug, pattern) in archive_specs.items():
        have = {ref for fam, ref in seen if fam == family}
        missing = required_months - have
        if not missing:
            continue
        archive_url = f"https://www.bls.gov/bls/news-release/{slug}.htm"
        asoup = BeautifulSoup(session.get(archive_url, timeout=30).text, "html.parser")
        for a in asoup.find_all("a"):
            href = str(a.get("href", ""))
            if f"/news.release/history/{slug}_" not in href or not href.lower().endswith(".txt"):
                continue
            release_date = release_date_from_archive_href(href)
            if release_date is None:
                continue
            estimated_ref = str(release_date.to_period("M") - 1)
            if estimated_ref not in missing:
                continue
            url = urljoin(archive_url, href)
            text = session.get(url, timeout=30).text[:12000]
            match = re.search(pattern, text, re.I)
            if not match:
                continue
            ref = f"{int(match.group(2)):04d}-{MONTHS[match.group(1).lower()]:02d}"
            if ref not in missing:
                continue
            rows.append({
                "family": family,
                "reference_month": ref,
                "release_date": release_date,
                "source_url": url,
            })
            seen.add((family, ref))
            missing.remove(ref)
            if not missing:
                break

    # Older JOLTS months are more complete in the release archive.
    archive_url = "https://www.bls.gov/bls/news-release/jolts.htm"
    soup = BeautifulSoup(session.get(archive_url, timeout=30).text, "html.parser")
    for a in soup.find_all("a"):
        href = str(a.get("href", ""))
        if "/news.release/history/jolts_" not in href or not href.lower().endswith(".txt"):
            continue
        release_date = release_date_from_archive_href(href)
        if release_date is None or not (2005 <= release_date.year <= 2025):
            continue
        url = urljoin(archive_url, href)
        text = session.get(url, timeout=30).text[:12000]
        m = re.search(
            r"JOB OPENINGS AND LABOR TURNOVER(?: SURVEY)?\s*[:\-]?\s*"
            r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})",
            text,
            re.I,
        )
        if not m:
            continue
        ref = f"{int(m.group(2)):04d}-{MONTHS[m.group(1).lower()]:02d}"
        key = ("JOLTS", ref)
        if key in seen:
            continue
        rows.append({"family": "JOLTS", "reference_month": ref, "release_date": release_date, "source_url": url})
        seen.add(key)

    out = pd.DataFrame(rows).sort_values(["family", "reference_month"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache, index=False, encoding="utf-8-sig")
    return out


def quarterly_release_from_monthly(calendar: pd.DataFrame, family: str) -> dict[str, pd.Timestamp]:
    d = calendar[calendar["family"] == family].copy()
    d["reference_month"] = pd.to_datetime(d["reference_month"] + "-01")
    d["quarter"] = d["reference_month"].dt.to_period("Q").astype(str)
    counts = d.groupby("quarter")["reference_month"].count()
    release = d.groupby("quarter")["release_date"].max()
    return {q: pd.Timestamp(dt).normalize() for q, dt in release.items() if counts.get(q, 0) == 3}


def monthly_release_map(calendar: pd.DataFrame, family: str) -> dict[str, pd.Timestamp]:
    d = calendar[calendar["family"] == family].copy()
    return {
        str(r.reference_month): pd.Timestamp(r.release_date).normalize()
        for r in d.itertuples(index=False)
    }


def quarterly_release_from_month_map(monthly: dict[str, pd.Timestamp]) -> dict[str, pd.Timestamp]:
    rows = []
    for ref, release_date in monthly.items():
        p = pd.Period(ref, freq="M")
        rows.append((str(p.asfreq("Q")), ref, release_date))
    out: dict[str, pd.Timestamp] = {}
    df = pd.DataFrame(rows, columns=["quarter", "reference_month", "release_date"])
    for quarter, g in df.groupby("quarter"):
        if len(g) == 3:
            out[str(quarter)] = pd.Timestamp(g["release_date"].max()).normalize()
    return out


def ppi_final_monthly_release(calendar: pd.DataFrame) -> dict[str, pd.Timestamp]:
    """Current unadjusted PPI values are exposed only when final (4-month revision cycle)."""
    initial = monthly_release_map(calendar, "PPI")
    out = {}
    for ref in initial:
        final_ref = str(pd.Period(ref, freq="M") + 4)
        if final_ref in initial:
            out[ref] = initial[final_ref]
    return out


def ces_benchmark_monthly_release(calendar: pd.DataFrame) -> dict[str, pd.Timestamp]:
    """Use current-final CES NSA values only after a safely completed benchmark cycle.

    National CES annual benchmarking can revise NSA observations over a span
    reaching from April of the prior year through December of the benchmark
    year. Using the January release two calendar years after a reference year is
    a conservative availability rule that never exposes today's benchmarked
    historical value before the relevant benchmark cycle has closed.
    """
    releases = monthly_release_map(calendar, "EMPLOYMENT_SITUATION")
    out = {}
    for ref in releases:
        year = int(ref[:4])
        benchmark_ref = f"{year + 2}-01"
        if benchmark_ref in releases:
            out[ref] = releases[benchmark_ref]
    return out


def jolts_final_monthly_release(calendar: pd.DataFrame) -> dict[str, pd.Timestamp]:
    """JOLTS current historical values become revision-safe after the 5-year annual revision window."""
    releases = monthly_release_map(calendar, "JOLTS")
    out = {}
    for ref in releases:
        year = int(ref[:4])
        final_ref = f"{year + 6}-01"
        if final_ref in releases:
            out[ref] = releases[final_ref]
    return out


def qcew_final_quarter_release(qcew_release: dict[str, pd.Timestamp]) -> dict[str, pd.Timestamp]:
    """BLS finalizes all quarters of year Y with the following year's Q1 full-data release."""
    out = {}
    for q in qcew_release:
        year = pd.Period(q, freq="Q").year
        final_ref = f"{year + 1}Q1"
        if final_ref in qcew_release:
            out[q] = qcew_release[final_ref]
    return out


def align_quarterly_panel_by_release(
    panel: pd.DataFrame | pd.Series,
    quarters: list[str],
    release_by_reference_quarter: dict[str, pd.Timestamp],
) -> pd.DataFrame | pd.Series:
    refs = [q for q in panel.index.astype(str) if q in release_by_reference_quarter]
    if isinstance(panel, pd.DataFrame):
        rows = []
        for origin in quarters:
            eligible = [q for q in refs if release_by_reference_quarter[q] <= quarter_end(origin)]
            if not eligible:
                rows.append(pd.Series(index=panel.columns, dtype=float, name=origin))
                continue
            latest = max(eligible, key=lambda q: pd.Period(q, freq="Q"))
            row = panel.loc[latest].copy()
            row.name = origin
            rows.append(row)
        return pd.DataFrame(rows, index=quarters, columns=panel.columns)
    values = []
    for origin in quarters:
        eligible = [q for q in refs if release_by_reference_quarter[q] <= quarter_end(origin)]
        if not eligible:
            values.append(np.nan)
            continue
        latest = max(eligible, key=lambda q: pd.Period(q, freq="Q"))
        values.append(panel.loc[latest])
    return pd.Series(values, index=quarters, name=panel.name, dtype=float)


def latest_reference_quarter(
    origin: str, release_map: dict[str, pd.Timestamp]
) -> str | None:
    cutoff = quarter_end(origin)
    eligible = [q for q, dt in release_map.items() if dt <= cutoff]
    if not eligible:
        return None
    return max(eligible, key=lambda q: pd.Period(q, freq="Q"))


CES_TYPES = {
    "01": "all_employees",
    "02": "avg_weekly_hours_all",
    "03": "avg_hourly_earnings_all",
    "06": "prod_nonsup_employees",
    "07": "avg_weekly_hours_prod_nonsup",
    "08": "avg_hourly_earnings_prod_nonsup",
    "10": "women_employees",
    "11": "avg_weekly_earnings_all",
    "16": "aggregate_weekly_hours_index",
    "17": "aggregate_weekly_payrolls_index",
}

JOLTS_ELEMENTS = {
    "JO": "job_openings",
    "HI": "hires",
    "QU": "quits",
    "LD": "layoffs_discharges",
    "TS": "total_separations",
    "OS": "other_separations",
}

GLOBAL_CPI = {
    "CUUR0000SA0": "cpi_all_items",
    "CUUR0000SA0L1E": "cpi_core",
    "CUUR0000SACL1E": "cpi_core_commodities",
    "CUUR0000SASLE": "cpi_core_services",
    "CUUR0000SAH1": "cpi_shelter",
    "CUUR0000SA0E": "cpi_energy",
    "CUUR0000SAF1": "cpi_food",
}

GLOBAL_CPS = {
    "LNU01000000": "civilian_labor_force",
    "LNU02000000": "civilian_employment",
    "LNU03000000": "civilian_unemployment",
    "LNU04000000": "unemployment_rate",
    "LNU01300000": "labor_force_participation",
    "LNU02300000": "employment_population_ratio",
    "LNU03327709": "u6_underutilization_rate",
}

GLOBAL_CES = {
    "CEU0000000001": "total_nonfarm_employment",
    "CEU0500000002": "private_avg_weekly_hours",
    "CEU0500000003": "private_avg_hourly_earnings",
    "CEU0500000016": "private_aggregate_weekly_hours_index",
    "CEU0500000017": "private_aggregate_weekly_payrolls_index",
}

GLOBAL_PPI = {
    "WPU00000000": "ppi_all_commodities",
    "WPUFD4": "ppi_final_demand",
    "WPUFD41": "ppi_final_demand_goods",
    "WPUFD42": "ppi_final_demand_services",
    "WPUFD49104": "ppi_final_demand_less_food_energy",
    "WPUID61": "ppi_processed_goods_intermediate",
    "WPUID62": "ppi_unprocessed_goods_intermediate",
}

MARKET_TICKERS = {
    "^GSPC": "sp500",
    "^DJI": "dow",
    "^IXIC": "nasdaq",
    "^RUT": "russell2000",
    "^VIX": "vix",
    "^IRX": "treasury_13w",
    "^FVX": "treasury_5y",
    "^TNX": "treasury_10y",
    "^TYX": "treasury_30y",
    "DX-Y.NYB": "dollar_index",
    "EURUSD=X": "eurusd",
    "JPY=X": "usdjpy",
    "GBPUSD=X": "gbpusd",
    "CL=F": "wti_crude",
    "NG=F": "natural_gas",
    "HG=F": "copper",
    "GC=F": "gold",
    "SI=F": "silver",
    "ZC=F": "corn",
    "ZW=F": "wheat",
    "XLE": "sector_energy",
    "XLB": "sector_materials",
    "XLI": "sector_industrials",
    "XLY": "sector_consumer_discretionary",
    "XLP": "sector_consumer_staples",
    "XLV": "sector_healthcare",
    "XLF": "sector_financials",
    "XLK": "sector_technology",
    "XLU": "sector_utilities",
    "IYR": "sector_real_estate",
    "VNQ": "reit_market",
    "HYG": "high_yield_credit",
    "LQD": "investment_grade_credit",
    "TLT": "long_treasury",
    "SHY": "short_treasury",
    "TIP": "tips",
    "XRT": "retail_industry",
    "IYT": "transportation_industry",
    "ITA": "aerospace_defense_industry",
    "SOXX": "semiconductor_industry",
    "IBB": "biotech_industry",
    "XHB": "homebuilders_industry",
    "KRE": "regional_banks_industry",
    "KBE": "banks_industry",
    "XME": "metals_mining_industry",
    "XOP": "oil_gas_exploration_industry",
    "OIH": "oil_services_industry",
    "XPH": "pharmaceutical_industry",
    "IAI": "broker_dealers_industry",
    "KIE": "insurance_industry",
}


def ensure_dirs() -> None:
    for p in (RAW_DIR, PROC_DIR, RESULTS_DIR):
        p.mkdir(parents=True, exist_ok=True)


def load_context() -> tuple[list[dict[str, str]], list[str]]:
    if not NODE_MAP.exists() or not TARGET_BEA_LONG.exists() or not TARGET_SUMMARY.exists():
        raise FileNotFoundError("Run 01_target_data_collection.py first")
    mapping = pd.read_csv(NODE_MAP, dtype=str).fillna("")
    names = (
        pd.read_csv(TARGET_BEA_LONG, usecols=["node_key", "node_name"], dtype=str)
        .drop_duplicates("node_key")
        .set_index("node_key")["node_name"]
        .to_dict()
    )
    nodes = []
    for row in mapping.to_dict("records"):
        nodes.append({
            "key": row["node_key"],
            "name": names[row["node_key"]],
            "naics_codes": row["qcew_naics"].replace(";", ","),
        })
    if len(nodes) != 56:
        raise RuntimeError(f"Expected 56 nodes, got {len(nodes)}")
    summary = json.loads(TARGET_SUMMARY.read_text(encoding="utf-8"))
    q0, q1 = summary["target_collection_period"]
    quarters = [f"{y}Q{q}" for y in range(int(q0[:4]), int(q1[:4]) + 1) for q in range(1, 5)]
    quarters = [q for q in quarters if q0 <= q <= q1]
    return nodes, quarters


def fetch_small_text(session: requests.Session, url: str, cache: Path, refresh: bool) -> str:
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists() and not refresh:
        return cache.read_text(encoding="utf-8", errors="replace")
    r = session.get(url, headers=BLS_HEADERS, timeout=90)
    r.raise_for_status()
    cache.write_bytes(r.content)
    return r.content.decode("utf-8", errors="replace")


def bls_filter_flat(
    session: requests.Session,
    url: str,
    wanted: set[str],
    cache: Path,
    refresh: bool,
) -> pd.DataFrame:
    """Stream a BLS flat file and retain only requested series IDs."""
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists() and not refresh:
        cached = pd.read_csv(cache)
        cached_ids = set(cached["series_id"].astype(str)) if "series_id" in cached else set()
        if wanted.issubset(cached_ids):
            return cached

    rows: list[tuple[str, int, str, float]] = []
    with session.get(url, headers=BLS_HEADERS, timeout=180, stream=True) as r:
        r.raise_for_status()
        for raw in r.iter_lines():
            if not raw:
                continue
            line = raw.decode("utf-8", errors="replace").strip("\r")
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            sid = parts[0].strip()
            if sid not in wanted:
                continue
            try:
                rows.append((sid, int(parts[1].strip()), parts[2].strip(), float(parts[3].strip())))
            except ValueError:
                continue
    df = pd.DataFrame(rows, columns=["series_id", "year", "period", "value"])
    df.to_csv(cache, index=False)
    return df


def quarterly_from_bls_rows(df: pd.DataFrame, quarters: list[str]) -> dict[str, pd.Series]:
    if df.empty:
        return {}
    d = df[df["period"].astype(str).str.fullmatch(r"M\d{2}")].copy()
    d["month"] = d["period"].str[1:].astype(int)
    d["quarter"] = d["year"].astype(str) + "Q" + (((d["month"] - 1) // 3) + 1).astype(str)
    out = {}
    for sid, g in d.groupby("series_id"):
        s = g.groupby("quarter")["value"].mean().reindex(quarters)
        s.index = quarters
        out[str(sid)] = s.astype(float)
    return out


def expand_compact_naics(spec: str) -> set[str]:
    """Expand BLS compact NAICS notation such as 3361,2,3 -> {3361,3362,3363}."""
    spec = str(spec).strip().replace(" ", "")
    if not spec or spec == "-":
        return set()
    if re.fullmatch(r"\d+-\d+", spec):
        return {spec}
    parts = [p for p in spec.split(",") if p]
    if not parts:
        return set()
    first = parts[0]
    out = {first}
    for token in parts[1:]:
        if token.isdigit() and first.isdigit() and len(token) < len(first):
            out.add(first[: len(first) - len(token)] + token)
        else:
            out.add(token)
    return out


def node_naics_set(node: dict[str, str]) -> set[str]:
    return {x.strip() for x in node["naics_codes"].split(",") if x.strip()}


def collect_qcew_employment(nodes: list[dict[str, str]], quarters: list[str]) -> dict[str, pd.DataFrame]:
    if not TARGET_EMPLOYMENT.exists():
        raise FileNotFoundError(TARGET_EMPLOYMENT)
    d = pd.read_csv(TARGET_EMPLOYMENT).set_index("quarter").reindex(quarters)
    d = d[[n["key"] for n in nodes]].apply(pd.to_numeric, errors="coerce")
    return {"qcew_employment": d}


def collect_ces_exact(
    session: requests.Session, nodes: list[dict[str, str]], quarters: list[str], refresh: bool
) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    meta_text = fetch_small_text(
        session,
        f"{BLS_ROOT}/ce/ce.series",
        RAW_DIR / "BLS" / "ce.series",
        refresh,
    )
    meta = pd.read_csv(io.StringIO(meta_text), sep="\t", dtype=str).fillna("")
    meta.columns = [c.strip() for c in meta.columns]
    for c in meta.columns:
        meta[c] = meta[c].astype(str).str.strip()
    valid = set(meta["series_id"])

    industry_text = fetch_small_text(
        session,
        f"{BLS_ROOT}/ce/ce.industry",
        RAW_DIR / "BLS" / "ce.industry",
        refresh,
    )
    industry = pd.read_csv(io.StringIO(industry_text), sep="\t", dtype=str).fillna("")
    industry.columns = [c.strip() for c in industry.columns]
    for c in industry.columns:
        industry[c] = industry[c].astype(str).str.strip()

    # Rebuild the node -> CES industry mapping from current BLS metadata rather
    # than relying on a previous audit artifact.
    node_industry: dict[str, str] = {}
    for node in nodes:
        target = node_naics_set(node)
        hits = industry[industry["naics_code"].map(lambda x: expand_compact_naics(x) == target)]
        if hits.empty:
            # A few CES publication groups use legacy supersector notation instead
            # of explicit NAICS despite having the same economic scope.
            fallback_names = {
                "retail": "Retail trade",
                "wholesale": "Wholesale trade",
                "information": "Information",
                "other_services": "Other services",
            }
            wanted_name = fallback_names.get(node["key"])
            if wanted_name:
                hits = industry[industry["industry_name"].str.casefold() == wanted_name.casefold()]
        if not hits.empty:
            # Prefer the most aggregate selectable row when duplicates exist.
            h = hits.copy()
            h["display_num"] = pd.to_numeric(h["display_level"], errors="coerce").fillna(99)
            node_industry[node["key"]] = str(h.sort_values("display_num").iloc[0]["industry_code"])

    sid_by_type_node: dict[str, dict[str, str]] = {dt: {} for dt in CES_TYPES}
    wanted = set(GLOBAL_CES)
    for dt in CES_TYPES:
        for node_key, industry_code in node_industry.items():
            sid = f"CEU{industry_code}{dt}"
            if sid in valid:
                sid_by_type_node[dt][node_key] = sid
                wanted.add(sid)

    rows = bls_filter_flat(
        session,
        f"{BLS_ROOT}/ce/ce.data.0.AllCESSeries",
        wanted,
        RAW_DIR / "BLS" / "ces_selected.csv",
        refresh,
    )
    qseries = quarterly_from_bls_rows(rows, quarters)
    node_keys = [n["key"] for n in nodes]
    out: dict[str, pd.DataFrame] = {}
    mapping_rows: list[dict] = []
    for dt, label in CES_TYPES.items():
        arr = np.full((len(quarters), len(nodes)), np.nan)
        for j, node in enumerate(nodes):
            sid = sid_by_type_node[dt].get(node["key"])
            if sid and sid in qseries:
                arr[:, j] = qseries[sid].values
                mapping_rows.append({"source": "BLS CES", "base_feature": f"ces_exact_{label}", "node_key": node["key"], "series_id": sid})
        df = pd.DataFrame(arr, index=quarters, columns=node_keys)
        if df.notna().any().any():
            out[f"ces_exact_{label}"] = df
    globals_out = {f"bls_{GLOBAL_CES[sid]}": qseries[sid] for sid in GLOBAL_CES if sid in qseries}
    return out, mapping_rows + [{"source": "GLOBAL_CES", "base_feature": k, "node_key": "ALL", "series_id": ""} for k in globals_out]


def collect_ppi_exact(
    session: requests.Session, nodes: list[dict[str, str]], quarters: list[str], refresh: bool
) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    meta_text = fetch_small_text(
        session,
        f"{BLS_ROOT}/pc/pc.series",
        RAW_DIR / "BLS" / "pc.series",
        refresh,
    )
    meta = pd.read_csv(io.StringIO(meta_text), sep="\t", dtype=str).fillna("")
    meta.columns = [c.strip() for c in meta.columns]
    for c in meta.columns:
        meta[c] = meta[c].astype(str).str.strip()

    def ppi_code_set(code: str) -> set[str]:
        code = str(code).strip().rstrip("-")
        return expand_compact_naics(code)

    # Industry net-output headline indexes have product_code == industry_code.
    headline = meta[
        (meta["industry_code"] == meta["product_code"])
        & meta["series_id"].str.startswith("PCU")
    ].copy()
    ppi_map: dict[str, str] = {}
    for node in nodes:
        target = node_naics_set(node)
        hits = headline[headline["industry_code"].map(lambda x: ppi_code_set(x) == target)]
        if hits.empty:
            continue
        hits = hits.copy()
        hits["begin_num"] = pd.to_numeric(hits["begin_year"], errors="coerce").fillna(9999)
        ppi_map[node["key"]] = str(hits.sort_values("begin_num").iloc[0]["series_id"])

    wanted = set(ppi_map.values())
    rows = bls_filter_flat(
        session,
        f"{BLS_ROOT}/pc/pc.data.0.Current",
        wanted,
        RAW_DIR / "BLS" / "ppi_industry_selected.csv",
        refresh,
    )
    qseries = quarterly_from_bls_rows(rows, quarters)
    arr = np.full((len(quarters), len(nodes)), np.nan)
    mapping = []
    for j, node in enumerate(nodes):
        sid = ppi_map.get(node["key"])
        if sid and sid in qseries:
            arr[:, j] = qseries[sid].values
            mapping.append({"source": "BLS PPI", "base_feature": "ppi_industry_net_output", "node_key": node["key"], "series_id": sid})
    return {"ppi_industry_net_output": pd.DataFrame(arr, index=quarters, columns=[n["key"] for n in nodes])}, mapping


def jolts_industry(node_key: str) -> str:
    mining = {"oil_gas_extraction", "mining_except_oil_gas", "support_mining"}
    durable = {
        "wood_products", "nonmetallic_mineral", "primary_metals", "fabricated_metal", "machinery",
        "computer_electronic", "electrical_equipment", "motor_vehicles", "other_transport_equipment",
        "furniture", "misc_manufacturing",
    }
    nondurable = {
        "food_beverage_tobacco", "textiles", "apparel_leather", "paper", "printing", "petroleum_coal",
        "chemicals", "plastics_rubber",
    }
    transport_util = {
        "utilities", "air_transport", "rail_transport", "water_transport", "truck_transport", "transit_ground",
        "pipeline", "other_transport_support", "warehousing",
    }
    finance = {"credit_intermediation", "securities", "insurance", "funds_trusts"}
    realestate = {"real_estate", "rental_leasing"}
    profbiz = {"legal", "computer_systems", "misc_professional", "management_companies", "admin_support", "waste_remediation"}
    health = {"ambulatory_health", "hospitals_nursing", "social_assistance"}
    arts = {"performing_arts_museums", "amusement_recreation"}
    if node_key in mining: return "110099"
    if node_key == "construction": return "230000"
    if node_key in durable: return "320000"
    if node_key in nondurable: return "340000"
    if node_key == "wholesale": return "420000"
    if node_key == "retail": return "440000"
    if node_key in transport_util: return "480099"
    if node_key == "information": return "510000"
    if node_key in finance: return "520000"
    if node_key in realestate: return "530000"
    if node_key in profbiz: return "540099"
    if node_key == "education": return "610000"
    if node_key in health: return "620000"
    if node_key in arts: return "710000"
    if node_key in {"accommodation", "food_services"}: return "720000"
    if node_key == "other_services": return "810000"
    raise KeyError(node_key)


def collect_jolts(
    session: requests.Session, nodes: list[dict[str, str]], quarters: list[str], refresh: bool
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.Series], list[dict]]:
    meta_text = fetch_small_text(session, f"{BLS_ROOT}/jt/jt.series", RAW_DIR / "BLS" / "jt.series", refresh)
    meta = pd.read_csv(io.StringIO(meta_text), sep="\t", dtype=str).fillna("")
    meta.columns = [c.strip() for c in meta.columns]
    for c in meta.columns:
        meta[c] = meta[c].astype(str).str.strip()
    meta = meta[(meta["seasonal"] == "U") & (meta["state_code"] == "00") & (meta["area_code"] == "00000") & (meta["sizeclass_code"] == "00")]

    node_industries = sorted({jolts_industry(n["key"]) for n in nodes})
    wanted_rows = meta[
        meta["industry_code"].isin(node_industries + ["000000"])
        & meta["dataelement_code"].isin(list(JOLTS_ELEMENTS) + ["UO"])
        & meta["ratelevel_code"].isin(["L", "R"])
    ]
    wanted = set(wanted_rows["series_id"])
    rows = bls_filter_flat(
        session,
        f"{BLS_ROOT}/jt/jt.data.1.AllItems",
        wanted,
        RAW_DIR / "BLS" / "jolts_selected.csv",
        refresh,
    )
    qseries = quarterly_from_bls_rows(rows, quarters)

    node_out: dict[str, pd.DataFrame] = {}
    mapping: list[dict] = []
    node_keys = [n["key"] for n in nodes]
    for elem, label in JOLTS_ELEMENTS.items():
        for rl, suffix in [("R", "rate"), ("L", "level")]:
            fname = f"jolts_{label}_{suffix}"
            arr = np.full((len(quarters), len(nodes)), np.nan)
            for j, node in enumerate(nodes):
                ind = jolts_industry(node["key"])
                hit = wanted_rows[(wanted_rows["industry_code"] == ind) & (wanted_rows["dataelement_code"] == elem) & (wanted_rows["ratelevel_code"] == rl)]
                if hit.empty:
                    continue
                sid = hit.iloc[0]["series_id"]
                if sid in qseries:
                    arr[:, j] = qseries[sid].values
                    mapping.append({"source": "BLS JOLTS", "base_feature": fname, "node_key": node["key"], "series_id": sid})
            node_out[fname] = pd.DataFrame(arr, index=quarters, columns=node_keys)

    global_out: dict[str, pd.Series] = {}
    total = wanted_rows[wanted_rows["industry_code"] == "000000"]
    for elem, label in {**JOLTS_ELEMENTS, "UO": "unemployed_per_opening"}.items():
        for rl, suffix in [("R", "rate"), ("L", "level")]:
            hit = total[(total["dataelement_code"] == elem) & (total["ratelevel_code"] == rl)]
            if hit.empty:
                continue
            sid = hit.iloc[0]["series_id"]
            if sid in qseries:
                global_out[f"bls_jolts_{label}_{suffix}"] = qseries[sid]
    return node_out, global_out, mapping


def collect_bls_global(
    session: requests.Session, quarters: list[str], refresh: bool
) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}

    cpi_rows = bls_filter_flat(
        session, f"{BLS_ROOT}/cu/cu.data.1.AllItems", set(GLOBAL_CPI), RAW_DIR / "BLS" / "cpi_selected.csv", refresh
    )
    for sid, s in quarterly_from_bls_rows(cpi_rows, quarters).items():
        out[f"bls_{GLOBAL_CPI[sid]}"] = s

    cps_rows = bls_filter_flat(
        session, f"{BLS_ROOT}/ln/ln.data.1.AllData", set(GLOBAL_CPS), RAW_DIR / "BLS" / "cps_selected.csv", refresh
    )
    for sid, s in quarterly_from_bls_rows(cps_rows, quarters).items():
        out[f"bls_{GLOBAL_CPS[sid]}"] = s

    ces_rows = bls_filter_flat(
        session, f"{BLS_ROOT}/ce/ce.data.0.AllCESSeries", set(GLOBAL_CES), RAW_DIR / "BLS" / "ces_global_selected.csv", refresh
    )
    for sid, s in quarterly_from_bls_rows(ces_rows, quarters).items():
        out[f"bls_{GLOBAL_CES[sid]}"] = s

    # Long-history all-commodities PPI.
    ppi_all = bls_filter_flat(
        session, f"{BLS_ROOT}/wp/wp.data.1.AllCommodities", {"WPU00000000"}, RAW_DIR / "BLS" / "ppi_global_all.csv", refresh
    )
    for sid, s in quarterly_from_bls_rows(ppi_all, quarters).items():
        out[f"bls_{GLOBAL_PPI[sid]}"] = s

    # Final-demand PPI begins in 2009; allowed because initial pool permits broad masked coverage.
    ppi_fd_ids = {sid for sid in GLOBAL_PPI if sid != "WPU00000000"}
    ppi_fd = bls_filter_flat(
        session, f"{BLS_ROOT}/wp/wp.data.0.Current", ppi_fd_ids, RAW_DIR / "BLS" / "ppi_global_fd.csv", refresh
    )
    for sid, s in quarterly_from_bls_rows(ppi_fd, quarters).items():
        out[f"bls_{GLOBAL_PPI[sid]}"] = s
    return out


def collect_nyfed(session: requests.Session, quarters: list[str], refresh: bool) -> dict[str, pd.Series]:
    cache = RAW_DIR / "NYFED" / "effr_2005_2024.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists() and not refresh:
        obj = json.loads(cache.read_text(encoding="utf-8"))
    else:
        params = {"startDate": "2005-01-01", "endDate": "2024-12-31", "type": "rate"}
        r = session.get(NYFED_EFFR, params=params, timeout=90)
        r.raise_for_status()
        obj = r.json()
        cache.write_text(json.dumps(obj), encoding="utf-8")
    d = pd.DataFrame(obj.get("refRates", []))
    if d.empty:
        return {}
    d["date"] = pd.to_datetime(d["effectiveDate"])
    d["quarter"] = d["date"].dt.to_period("Q").astype(str)
    out = {}
    for col, name in [("percentRate", "nyfed_effr"), ("targetRateFrom", "nyfed_target_lower"), ("targetRateTo", "nyfed_target_upper")]:
        if col not in d:
            continue
        d[col] = pd.to_numeric(d[col], errors="coerce")
        s = d.groupby("quarter")[col].mean().reindex(quarters)
        s.index = quarters
        out[name] = s
    return out


def collect_market(quarters: list[str], refresh: bool) -> tuple[dict[str, pd.Series], list[dict]]:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance is required for market-context features") from exc

    # Use raw closes, never retrospectively dividend/split-adjusted prices.
    cache = RAW_DIR / "MARKET" / "daily_market_close_unadjusted.csv"
    cache.parent.mkdir(parents=True, exist_ok=True)
    use_cache = cache.exists() and not refresh
    if use_cache:
        daily = pd.read_csv(cache, header=[0, 1], index_col=0, parse_dates=True)
        cached_tickers = set(daily.columns.get_level_values(1)) if isinstance(daily.columns, pd.MultiIndex) else set(daily.columns)
        use_cache = set(MARKET_TICKERS).issubset(cached_tickers)
    if not use_cache:
        data = yf.download(
            list(MARKET_TICKERS), start="2005-01-01", end="2025-01-06",
            progress=False, auto_adjust=False, threads=True, group_by="column",
        )
        if data.empty:
            raise RuntimeError("Yahoo Finance returned no market data")
        if isinstance(data.columns, pd.MultiIndex):
            price_key = "Close"
            daily = data[price_key].copy()
        else:
            price_key = "Close" if "Close" in data.columns else "Adj Close"
            daily = data[[price_key]].rename(columns={price_key: list(MARKET_TICKERS)[0]})
        daily.columns = pd.MultiIndex.from_product([["price"], daily.columns])
        daily.to_csv(cache)

    # Normalize loaded representation to columns=tickers.
    if isinstance(daily.columns, pd.MultiIndex):
        if "price" in daily.columns.get_level_values(0):
            px = daily["price"].copy()
        elif "Close" in daily.columns.get_level_values(0):
            px = daily["Close"].copy()
        else:
            px = daily.xs(daily.columns.get_level_values(0)[0], level=0, axis=1).copy()
    else:
        px = daily.copy()
    px.index = pd.to_datetime(px.index)

    out: dict[str, pd.Series] = {}
    audit: list[dict] = []
    qindex = pd.PeriodIndex(quarters, freq="Q")
    for ticker, label in MARKET_TICKERS.items():
        if ticker not in px.columns:
            audit.append({"ticker": ticker, "label": label, "status": "MISSING"})
            continue
        s = pd.to_numeric(px[ticker], errors="coerce").dropna()
        if s.empty:
            audit.append({"ticker": ticker, "label": label, "status": "EMPTY"})
            continue
        tmp = pd.DataFrame({"price": s})
        tmp["q"] = tmp.index.to_period("Q")
        mean = tmp.groupby("q")["price"].mean().reindex(qindex)
        qret = tmp.groupby("q")["price"].apply(lambda x: float(np.log(x.iloc[-1] / x.iloc[0])) if len(x) >= 2 and x.iloc[0] > 0 and x.iloc[-1] > 0 else np.nan).reindex(qindex)
        prev = s.shift(1)
        ratio = (s / prev).where((s > 0) & (prev > 0))
        daily_ret = np.log(ratio)
        vol_df = pd.DataFrame({"r": daily_ret}).dropna()
        vol_df["q"] = vol_df.index.to_period("Q")
        vol = (vol_df.groupby("q")["r"].std() * math.sqrt(252)).reindex(qindex)
        for metric, ser in [("level", mean), ("qret", qret), ("vol", vol)]:
            ser.index = quarters
            out[f"market_{label}_{metric}"] = ser
        audit.append({"ticker": ticker, "label": label, "status": "PASS", "start": str(s.index.min().date()), "end": str(s.index.max().date())})
    return out, audit


def node_stats(df: pd.DataFrame) -> tuple[float, float, float]:
    per_node = df.notna().mean(axis=0)
    observed = per_node[per_node > 0]
    node_fraction = float((per_node > 0).mean())
    median_time = float(observed.median()) if not observed.empty else 0.0
    mean_all = float(df.notna().mean().mean())
    return node_fraction, median_time, mean_all


def add_derivatives(base: pd.DataFrame | pd.Series, prefix: str) -> dict[str, pd.DataFrame | pd.Series]:
    out = {prefix: base, f"{prefix}__d1": base.diff(1), f"{prefix}__d4": base.diff(4)}
    vals = base.to_numpy(dtype=float) if isinstance(base, pd.DataFrame) else base.to_numpy(dtype=float)
    finite = vals[np.isfinite(vals)]
    if finite.size and np.all(finite > 0):
        logx = np.log(base)
        out[f"{prefix}__logd1"] = logx.diff(1)
        out[f"{prefix}__logd4"] = logx.diff(4)
    return out


def expand_node(
    blocks: dict[str, pd.DataFrame], lag: int, source: str, group: str, registry: list[dict]
) -> dict[str, pd.DataFrame]:
    out = {}
    for base_name, df in blocks.items():
        shifted = df.shift(lag)
        node_fraction, median_time, mean_all = node_stats(shifted)
        base_ok = node_fraction >= MIN_NODE_FRACTION and median_time >= MIN_TIME_COVERAGE
        for fid, obj in add_derivatives(shifted, base_name).items():
            nf, mt, ma = node_stats(obj)
            status = "PASS" if base_ok else "COVERAGE_FAIL"
            registry.append({
                "feature_id": fid, "base_feature": base_name, "scope": "NODE", "source": source, "group": group,
                "safe_lag_quarters": lag, "node_fraction": nf, "median_time_coverage": mt, "overall_coverage": ma,
                "status": status,
                "target_overlap_control": "TARGET_INGREDIENT_LAGGED_ONLY" if base_name == "qcew_employment" else "NONE",
            })
            if status == "PASS":
                out[fid] = obj
    return out


def expand_global(
    blocks: dict[str, pd.Series], lag: int, source: str, group: str, registry: list[dict]
) -> dict[str, pd.Series]:
    out = {}
    for base_name, s in blocks.items():
        shifted = s.shift(lag)
        base_cov = float(shifted.notna().mean())
        base_ok = base_cov >= MIN_TIME_COVERAGE
        for fid, obj in add_derivatives(shifted, base_name).items():
            cov = float(obj.notna().mean())
            status = "PASS" if base_ok else "COVERAGE_FAIL"
            registry.append({
                "feature_id": fid, "base_feature": base_name, "scope": "GLOBAL", "source": source, "group": group,
                "safe_lag_quarters": lag, "node_fraction": 1.0, "median_time_coverage": cov, "overall_coverage": cov,
                "status": status, "target_overlap_control": "NONE",
            })
            if status == "PASS":
                out[fid] = obj
    return out


def write_node_long(path: Path, quarters: list[str], nodes: list[dict[str, str]], features: dict[str, pd.DataFrame]) -> None:
    rows = []
    for t, q in enumerate(quarters):
        for j, node in enumerate(nodes):
            row = {"quarter": q, "node_key": node["key"]}
            for fid, df in features.items():
                v = df.iloc[t, j]
                row[fid] = "" if pd.isna(v) else float(v)
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def write_node_masks(path: Path, quarters: list[str], nodes: list[dict[str, str]], features: dict[str, pd.DataFrame]) -> None:
    rows = []
    for t, q in enumerate(quarters):
        for j, node in enumerate(nodes):
            row = {"quarter": q, "node_key": node["key"]}
            for fid, df in features.items():
                row[fid] = int(pd.notna(df.iloc[t, j]))
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def collect(refresh: bool) -> None:
    ensure_dirs()
    nodes, quarters = load_context()
    session = requests.Session()
    session.headers.update(BLS_HEADERS)

    print("[0/8] Official release-date calendars")
    bls_calendar = build_bls_monthly_calendar(session, refresh=refresh)
    target_release = pd.read_csv(TARGET_RELEASE_METADATA)
    qcew_calendar = target_release[target_release["family"] == "QCEW_QUARTERLY"].copy()
    qcew_calendar = qcew_calendar.rename(columns={"reference_period": "reference_quarter"})
    qcew_calendar["release_date"] = pd.to_datetime(qcew_calendar["release_date"])
    qcew_release_raw = {
        str(r.reference_quarter): pd.Timestamp(r.release_date).normalize()
        for r in qcew_calendar.itertuples(index=False)
    }
    release_maps = {
        # CPI-U/W unadjusted indexes are final when issued; unadjusted CPS source
        # data are not subject to the rolling 5-year seasonal-factor revisions.
        "CPI_FINAL": quarterly_release_from_monthly(bls_calendar, "CPI"),
        "CPS_FINAL": quarterly_release_from_monthly(bls_calendar, "EMPLOYMENT_SITUATION"),
        # Current-final CES/PPI/JOLTS/QCEW values are exposed only after their
        # official revision windows have closed.
        "CES_FINAL": quarterly_release_from_month_map(ces_benchmark_monthly_release(bls_calendar)),
        "PPI_FINAL": quarterly_release_from_month_map(ppi_final_monthly_release(bls_calendar)),
        "JOLTS_FINAL": quarterly_release_from_month_map(jolts_final_monthly_release(bls_calendar)),
        "QCEW_FINAL": qcew_final_quarter_release(qcew_release_raw),
        "MARKET": {q: quarter_end(q) for q in quarters},
        "NYFED_EFFR": {q: quarter_end(q) + FED_BUSINESS_DAY for q in quarters},
        "NYFED_TARGET": {q: quarter_end(q) for q in quarters},
    }

    print("[1/8] QCEW lagged employment")
    qcew = collect_qcew_employment(nodes, quarters)

    print("[2/8] CES exact industry labor features")
    ces, ces_mapping = collect_ces_exact(session, nodes, quarters, refresh)

    print("[3/8] Industry PPI")
    ppi, ppi_mapping = collect_ppi_exact(session, nodes, quarters, refresh)

    print("[4/8] JOLTS industry labor-demand / turnover features")
    jolts_node, jolts_global, jolts_mapping = collect_jolts(session, nodes, quarters, refresh)

    print("[5/8] National BLS + NY Fed context")
    bls_global = collect_bls_global(session, quarters, refresh)
    nyfed = collect_nyfed(session, quarters, refresh)

    # BLS publishes unemployed-per-opening directly only as a seasonally adjusted
    # JOLTS ratio. Reconstruct the same concept from revision-safe unadjusted
    # components at the same reference quarter so the base-feature universe stays
    # comparable without reintroducing seasonal-factor revision leakage.
    if (
        "bls_civilian_unemployment" in bls_global
        and "bls_jolts_job_openings_level" in jolts_global
    ):
        openings = jolts_global["bls_jolts_job_openings_level"].replace(0.0, np.nan)
        unemployed = bls_global["bls_civilian_unemployment"]
        jolts_global["bls_jolts_unemployed_per_opening_rate"] = unemployed / openings

    print("[6/8] Market / FX / commodity context")
    market, market_audit = collect_market(quarters, refresh)

    # Convert each reference-quarter panel to the latest quarter that had
    # actually been published by the model's quarter-end forecast origin.
    qcew = align_blocks(qcew, quarters, release_maps["QCEW_FINAL"])
    ces = align_blocks(ces, quarters, release_maps["CES_FINAL"])
    ppi = align_blocks(ppi, quarters, release_maps["PPI_FINAL"])
    jolts_node = align_blocks(jolts_node, quarters, release_maps["JOLTS_FINAL"])
    jolts_global = align_blocks(jolts_global, quarters, release_maps["JOLTS_FINAL"])

    cpi_names = {f"bls_{v}" for v in GLOBAL_CPI.values()}
    cps_names = {f"bls_{v}" for v in GLOBAL_CPS.values()}
    ces_global_names = {f"bls_{v}" for v in GLOBAL_CES.values()}
    ppi_global_names = {f"bls_{v}" for v in GLOBAL_PPI.values()}
    bls_aligned: dict[str, pd.Series] = {}
    for name, series in bls_global.items():
        if name in cpi_names:
            rel = release_maps["CPI_FINAL"]
        elif name in cps_names:
            rel = release_maps["CPS_FINAL"]
        elif name in ces_global_names:
            rel = release_maps["CES_FINAL"]
        elif name in ppi_global_names:
            rel = release_maps["PPI_FINAL"]
        else:
            raise RuntimeError(f"No release calendar family for BLS feature {name}")
        bls_aligned[name] = align_quarterly_panel_by_release(series, quarters, rel)
    bls_global = bls_aligned

    nyfed_aligned: dict[str, pd.Series] = {}
    for name, series in nyfed.items():
        rel = release_maps["NYFED_EFFR"] if name == "nyfed_effr" else release_maps["NYFED_TARGET"]
        nyfed_aligned[name] = align_quarterly_panel_by_release(series, quarters, rel)
    nyfed = nyfed_aligned
    market = align_blocks(market, quarters, release_maps["MARKET"])

    release_audit_rows = []
    release_metadata_rows = []
    for family, rel in release_maps.items():
        for ref, release_date in sorted(rel.items()):
            release_metadata_rows.append({
                "family": family,
                "reference_period": ref,
                "release_date": pd.Timestamp(release_date).date().isoformat(),
                "source": (
                    "BLS" if family in {"CPI_FINAL", "CPS_FINAL", "CES_FINAL", "PPI_FINAL", "JOLTS_FINAL", "QCEW_FINAL"}
                    else "NY Fed" if family.startswith("NYFED")
                    else "Market"
                ),
            })
        for origin in quarters:
            ref = latest_reference_quarter(origin, rel)
            release_audit_rows.append({
                "family": family,
                "forecast_origin_quarter": origin,
                "forecast_origin_date": quarter_end(origin).date().isoformat(),
                "latest_reference_quarter": ref,
                "latest_reference_release_date": (
                    rel[ref].date().isoformat() if ref is not None else ""
                ),
            })
    pd.DataFrame(release_audit_rows).to_csv(
        RESULTS_DIR / "release_cutoff_audit.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(release_metadata_rows).drop_duplicates(
        ["family", "reference_period"]
    ).to_csv(FEATURE_RELEASE_METADATA, index=False, encoding="utf-8-sig")

    registry: list[dict] = []
    node_features: dict[str, pd.DataFrame] = {}
    node_features.update(expand_node(qcew, 0, "BLS QCEW", "NODE_LABOR", registry))
    node_features.update(expand_node(ces, 0, "BLS CES", "NODE_LABOR", registry))
    node_features.update(expand_node(ppi, 0, "BLS PPI", "NODE_PRICE", registry))
    node_features.update(expand_node(jolts_node, 0, "BLS JOLTS", "NODE_LABOR_DEMAND", registry))

    global_features: dict[str, pd.Series] = {}
    global_features.update(expand_global(bls_global, 0, "BLS", "MACRO_PRICE_LABOR", registry))
    global_features.update(expand_global(jolts_global, 0, "BLS JOLTS", "MACRO_LABOR_DEMAND", registry))
    global_features.update(expand_global(nyfed, 0, "NY Fed", "MONETARY_POLICY", registry))
    global_features.update(expand_global(market, 0, "Yahoo Finance", "MARKET_FINANCIAL_COMMODITY", registry))

    print("[7/8] Writing panels / masks / audit")
    write_node_long(PROC_DIR / "node_features_quarterly_long.csv", quarters, nodes, node_features)
    write_node_masks(PROC_DIR / "node_feature_masks_quarterly_long.csv", quarters, nodes, node_features)
    pd.DataFrame({"quarter": quarters, **{k: v.values for k, v in global_features.items()}}).to_csv(
        PROC_DIR / "global_features_quarterly.csv", index=False, encoding="utf-8-sig"
    )

    reg = pd.DataFrame(registry)
    reg.to_csv(RESULTS_DIR / "initial_feature_registry.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(ces_mapping + ppi_mapping + jolts_mapping).to_csv(
        RESULTS_DIR / "node_source_mapping.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(market_audit).to_csv(RESULTS_DIR / "market_source_audit.csv", index=False, encoding="utf-8-sig")

    base_rows = (
        [{"base_feature": k, "scope": "NODE", "source": "BLS QCEW"} for k in qcew]
        + [{"base_feature": k, "scope": "NODE", "source": "BLS CES"} for k in ces]
        + [{"base_feature": k, "scope": "NODE", "source": "BLS PPI"} for k in ppi]
        + [{"base_feature": k, "scope": "NODE", "source": "BLS JOLTS"} for k in jolts_node]
        + [{"base_feature": k, "scope": "GLOBAL", "source": "BLS"} for k in bls_global]
        + [{"base_feature": k, "scope": "GLOBAL", "source": "BLS JOLTS"} for k in jolts_global]
        + [{"base_feature": k, "scope": "GLOBAL", "source": "NY Fed"} for k in nyfed]
        + [{"base_feature": k, "scope": "GLOBAL", "source": "Yahoo Finance"} for k in market]
    )
    pd.DataFrame(base_rows).drop_duplicates().to_csv(
        RESULTS_DIR / "initial_base_feature_set.csv", index=False, encoding="utf-8-sig"
    )

    research = """# PRISM initial feature pool — research basis

## Selection philosophy
The ICI target is defined conservatively. Features are collected broadly first and are
not restricted to official statistics when a reproducible, economically meaningful
market proxy exists. Final retention is deferred to train-only selection, regularization,
and ablation.

## Source rationale
- BLS QCEW: exact industry employment; target ingredient is exposed only with a conservative lag.
- BLS CES: monthly industry employment, hours, earnings, payroll and labor-composition measures.
- BLS PPI: industry net-output prices, i.e. prices received for output sold outside the industry.
- BLS JOLTS: job openings, hires, quits, layoffs/discharges and separations as labor-demand/turnover signals.
- BLS CPI/CPS/CES/PPI national series: common inflation, labor-slack and activity context.
- NY Fed EFFR: observed monetary-policy implementation rate.
- Yahoo Finance: reproducible observed market prices for equities, rates proxies, FX, energy,
  industrial metals, precious metals and broad sector ETFs. These are treated as predictive
  market context, not as authoritative definitions of industry state.

## High-dimensional-predictor rationale
FRED-MD and the Stock-Watson many-predictor literature explicitly motivate broad information
sets followed by factor extraction / variable selection rather than narrow hand-picking.
PRISM follows that philosophy at the candidate stage while preserving out-of-sample selection.

## Timing / leakage
- Fixed quarter lags are not used.
- Every predictor is admitted only when its revision-safe availability date is on or before the quarter-end forecast origin.
- CPI and CPS use unadjusted series; CPI-U/W unadjusted observations are final when issued.
- CES uses unadjusted current-final observations only after the following annual benchmark release.
- PPI uses unadjusted current-final observations only after the four-month revision window closes.
- JOLTS uses unadjusted current-final observations only after the five-year annual revision window closes.
- QCEW current-final observations are exposed only after the following year's Q1 full-data release finalizes that calendar year.
- NY Fed EFFR follows its next-business-day publication rule; market data are available through the quarter-end close.
- Market predictors use raw Close, never retrospectively adjusted Close.
- Scaling and final feature selection are deferred and must be fit on training data only.

## Coverage policy
Structural non-publication for a node is retained as an explicit mask when the same economic
concept is published for a substantial subset of industries. This deliberately replaces the old
"56/56 or exclude" rule.
"""
    (RESULTS_DIR / "research_basis.md").write_text(research, encoding="utf-8")

    passed = reg[reg["status"] == "PASS"]
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "period": [quarters[0], quarters[-1]],
        "quarters": len(quarters),
        "nodes": len(nodes),
        "base_feature_candidates": len(pd.DataFrame(base_rows).drop_duplicates()),
        "node_base_candidates": len(qcew) + len(ces) + len(ppi) + len(jolts_node),
        "global_base_candidates": len(bls_global) + len(jolts_global) + len(nyfed) + len(market),
        "node_model_channels_pass": len(node_features),
        "global_model_channels_pass": len(global_features),
        "total_initial_model_channels": len(node_features) + len(global_features),
        "minimum_time_coverage": MIN_TIME_COVERAGE,
        "minimum_node_fraction_for_partial_node_feature": MIN_NODE_FRACTION,
        "registry_pass_rows": int(len(passed)),
        "availability_alignment": "point-in-time/revision-safe: unadjusted final series where possible; finalization-gated CES/PPI/JOLTS/QCEW; raw market Close; no fixed quarter lags",
        "status": "PASS",
    }
    (RESULTS_DIR / "feature_pool_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[8/8] PASS")
    print(json.dumps(summary, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(description="Build broad initial PRISM feature pool")
    p.add_argument("--refresh", action="store_true", help="redownload source data")
    args = p.parse_args()
    collect(args.refresh)


if __name__ == "__main__":
    main()
