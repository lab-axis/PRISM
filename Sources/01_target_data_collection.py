#!/usr/bin/env python3
"""Collect every official input required to construct PRISM's quarterly ICI target.

Collected sources
-----------------
BEA GDP-by-Industry workbooks
  - Current Value Added       : TVA105-Q
  - Real Value Added          : TVA106-Q
  - Current Gross Output      : TGO105-Q
  - Real Gross Output         : TGO106-Q
  - Gross Operating Surplus   : TVA113-A (annual)

BLS QCEW
  - Monthly employment for every NAICS code used by the fixed 56-node registry.

The script downloads raw source files, aligns them to the 56 PRISM nodes, validates
complete coverage over the target construction period, and writes tidy CSV files.
It intentionally does NOT calculate GOS temporal disaggregation, component scores,
or ICI; those belong to the target-construction stage, not source collection.
"""

from __future__ import annotations

import argparse
import io
import itertools
import pickle
import csv
import hashlib
import json
import re
import time
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

try:
    import xlrd
except ImportError as exc:  # pragma: no cover - explicit environment failure
    raise RuntimeError("stage 01 point-in-time history requires xlrd>=2.0 for BEA .xls vintages") from exc


ROOT = Path(__file__).resolve().parent.parent
JOB = "01_target_data_collection"
DATA_DIR = ROOT / "Data" / JOB
RAW_BEA_DIR = DATA_DIR / "raw" / "BEA"
RAW_BLS_DIR = DATA_DIR / "raw" / "BLS"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT / "Results" / JOB
NODE_FILE = RESULTS_DIR / "node_registry_56.csv"

BEA_VALUE_ADDED_URL = "https://apps.bea.gov/industry/Release/XLS/GDPxInd/ValueAdded.xlsx"
BEA_GROSS_OUTPUT_URL = "https://apps.bea.gov/industry/Release/XLS/GDPxInd/GrossOutput.xlsx"
BLS_API_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
BLS_CES_FLAT_URL = "https://download.bls.gov/pub/time.series/ce/ce.data.0.AllCESSeries"
RAIL_CES_SERIES = "CEU4348200001"  # Rail transportation, all employees, NSA, thousands.
FUNDS_TRUSTS_CES_SERIES = "CEU5552300001"  # Securities/investments + funds/trusts aggregate, all employees, NSA, thousands.

TARGET_START_YEAR = 2005
HISTORY_START_YEAR = 2003
EXPECTED_NODE_COUNT = 56

RELEASE_METADATA_CSV = PROCESSED_DIR / "target_release_metadata.csv"
PIT_HISTORY_RAW_CSV = PROCESSED_DIR / "point_in_time_history_raw.csv"
PIT_HISTORY_AUDIT_CSV = RESULTS_DIR / "point_in_time_history_audit.csv"

MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}
QWORDS = {
    "first": 1, "1st": 1, "second": 2, "2nd": 2,
    "third": 3, "3rd": 3, "fourth": 4, "4th": 4,
}


# BEA labels that differ from the node-registry presentation labels.
BEA_ALIAS = {
    "credit_intermediation": "Federal Reserve banks, credit intermediation, and related activities",
    "misc_professional": "Miscellaneous professional, scientific, and technical services",
    "rental_leasing": "Rental and leasing services and lessors of intangible assets",
    "education": "Educational services",
    "performing_arts_museums": "Performing arts, spectator sports, museums, and related activities",
    "amusement_recreation": "Amusements, gambling, and recreation industries",
    "motor_vehicles": "Motor vehicles, bodies and trailers, and parts",
    "food_beverage_tobacco": "Food and beverage and tobacco products",
    "apparel_leather": "Apparel and leather and allied products",
    "other_services": "Other services, except government",
    "mining_except_oil_gas": "Mining, except oil and gas",
    "electrical_equipment": "Electrical equipment, appliances, and components",
    "securities": "Securities, commodity contracts, and investments",
    "funds_trusts": "Funds, trusts, and other financial vehicles",
}


# Canonical PRISM industry universe. Stage 01 owns this registry so a clean
# checkout can run from scratch without depending on an archived pilot/audit
# stage or on outputs from a previous Stage-01 run.
CANONICAL_NODES = [
    {"key": "oil_gas_extraction", "name": "Oil and gas extraction", "naics_codes": "211"},
    {"key": "mining_except_oil_gas", "name": "Mining, except oil and gas", "naics_codes": "212"},
    {"key": "support_mining", "name": "Support activities for mining", "naics_codes": "213"},
    {"key": "utilities", "name": "Utilities", "naics_codes": "22"},
    {"key": "construction", "name": "Construction", "naics_codes": "23"},
    {"key": "wood_products", "name": "Wood products", "naics_codes": "321"},
    {"key": "nonmetallic_mineral", "name": "Nonmetallic mineral products", "naics_codes": "327"},
    {"key": "primary_metals", "name": "Primary metals", "naics_codes": "331"},
    {"key": "fabricated_metal", "name": "Fabricated metal products", "naics_codes": "332"},
    {"key": "machinery", "name": "Machinery", "naics_codes": "333"},
    {"key": "computer_electronic", "name": "Computer and electronic products", "naics_codes": "334"},
    {"key": "electrical_equipment", "name": "Electrical equipment, appliances, and components", "naics_codes": "335"},
    {"key": "motor_vehicles", "name": "Motor vehicles, bodies and trailers, and parts", "naics_codes": "3361,3362,3363"},
    {"key": "other_transport_equipment", "name": "Other transportation equipment", "naics_codes": "3364,3365,3366,3369"},
    {"key": "furniture", "name": "Furniture and related products", "naics_codes": "337"},
    {"key": "misc_manufacturing", "name": "Miscellaneous manufacturing", "naics_codes": "339"},
    {"key": "food_beverage_tobacco", "name": "Food and beverage and tobacco products", "naics_codes": "311,312"},
    {"key": "textiles", "name": "Textile mills and textile product mills", "naics_codes": "313,314"},
    {"key": "apparel_leather", "name": "Apparel and leather and allied products", "naics_codes": "315,316"},
    {"key": "paper", "name": "Paper products", "naics_codes": "322"},
    {"key": "printing", "name": "Printing and related support activities", "naics_codes": "323"},
    {"key": "petroleum_coal", "name": "Petroleum and coal products", "naics_codes": "324"},
    {"key": "chemicals", "name": "Chemical products", "naics_codes": "325"},
    {"key": "plastics_rubber", "name": "Plastics and rubber products", "naics_codes": "326"},
    {"key": "wholesale", "name": "Wholesale trade", "naics_codes": "42"},
    {"key": "retail", "name": "Retail trade", "naics_codes": "44-45"},
    {"key": "air_transport", "name": "Air transportation", "naics_codes": "481"},
    {"key": "rail_transport", "name": "Rail transportation", "naics_codes": "482"},
    {"key": "water_transport", "name": "Water transportation", "naics_codes": "483"},
    {"key": "truck_transport", "name": "Truck transportation", "naics_codes": "484"},
    {"key": "transit_ground", "name": "Transit and ground passenger transportation", "naics_codes": "485"},
    {"key": "pipeline", "name": "Pipeline transportation", "naics_codes": "486"},
    {"key": "other_transport_support", "name": "Other transportation and support activities", "naics_codes": "487,488"},
    {"key": "warehousing", "name": "Warehousing and storage", "naics_codes": "493"},
    {"key": "information", "name": "Information", "naics_codes": "51"},
    {"key": "credit_intermediation", "name": "Federal Reserve banks, credit intermediation, and related activities", "naics_codes": "521,522"},
    {"key": "securities", "name": "Securities, commodity contracts, and investments", "naics_codes": "523"},
    {"key": "insurance", "name": "Insurance carriers and related activities", "naics_codes": "524"},
    {"key": "funds_trusts", "name": "Funds, trusts, and other financial vehicles", "naics_codes": "525"},
    {"key": "real_estate", "name": "Real estate", "naics_codes": "531"},
    {"key": "rental_leasing", "name": "Rental and leasing services and lessors of intangible assets", "naics_codes": "532,533"},
    {"key": "legal", "name": "Legal services", "naics_codes": "5411"},
    {"key": "computer_systems", "name": "Computer systems design and related services", "naics_codes": "5415"},
    {"key": "misc_professional", "name": "Miscellaneous professional, scientific, and technical services", "naics_codes": "5412,5413,5414,5416,5417,5418,5419"},
    {"key": "management_companies", "name": "Management of companies and enterprises", "naics_codes": "55"},
    {"key": "admin_support", "name": "Administrative and support services", "naics_codes": "561"},
    {"key": "waste_remediation", "name": "Waste management and remediation services", "naics_codes": "562"},
    {"key": "education", "name": "Educational services", "naics_codes": "61"},
    {"key": "ambulatory_health", "name": "Ambulatory health care services", "naics_codes": "621"},
    {"key": "hospitals_nursing", "name": "Hospitals, nursing and residential care facilities", "naics_codes": "622,623"},
    {"key": "social_assistance", "name": "Social assistance", "naics_codes": "624"},
    {"key": "performing_arts_museums", "name": "Performing arts, spectator sports, museums, and related activities", "naics_codes": "711,712"},
    {"key": "amusement_recreation", "name": "Amusements, gambling, and recreation industries", "naics_codes": "713"},
    {"key": "accommodation", "name": "Accommodation", "naics_codes": "721"},
    {"key": "food_services", "name": "Food services and drinking places", "naics_codes": "722"},
    {"key": "other_services", "name": "Other services, except government", "naics_codes": "81"},
]


def ensure_dirs() -> None:
    for path in (RAW_BEA_DIR, RAW_BLS_DIR, PROCESSED_DIR, RESULTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(session: requests.Session, url: str, path: Path, refresh: bool) -> dict[str, str | int]:
    if refresh or not path.exists():
        tmp = path.with_suffix(path.suffix + ".part")
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                with session.get(url, timeout=120, stream=True) as response:
                    response.raise_for_status()
                    with tmp.open("wb") as f:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)
                tmp.replace(path)
                last_error = None
                break
            except Exception as exc:  # network retry only
                last_error = exc
                if tmp.exists():
                    tmp.unlink()
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        if last_error is not None:
            raise last_error
    return {
        "url": url,
        "path": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def load_nodes() -> list[dict[str, str]]:
    nodes = [dict(row) for row in CANONICAL_NODES]
    if len(nodes) != EXPECTED_NODE_COUNT:
        raise RuntimeError(f"Expected {EXPECTED_NODE_COUNT} nodes, found {len(nodes)}")
    keys = [row["key"] for row in nodes]
    if len(set(keys)) != EXPECTED_NODE_COUNT:
        raise RuntimeError("Canonical 56-node registry contains duplicate keys")
    NODE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with NODE_FILE.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["key", "name", "naics_codes"])
        writer.writeheader()
        writer.writerows(nodes)
    return nodes


def fetch_ces_employment_series(
    session: requests.Session,
    series_id: str,
    cache_stem: str,
    end_year: int,
    refresh: bool,
) -> tuple[list[str], np.ndarray, Path]:
    """Fetch one official CES NSA all-employees series and return monthly persons."""
    cache = RAW_BLS_DIR / f"{cache_stem}_{HISTORY_START_YEAR}_{end_year}.csv"
    if cache.exists() and not refresh:
        rows = list(csv.DictReader(cache.open(encoding="utf-8-sig", newline="")))
    else:
        rows: list[dict[str, str]] = []
        with session.get(
            BLS_CES_FLAT_URL,
            headers={"User-Agent": "PRISM academic research haseung.ryu.ai@gmail.com"},
            timeout=180,
            stream=True,
        ) as response:
            response.raise_for_status()
            for raw in response.iter_lines():
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="replace").strip("\r")
                parts = [x.strip() for x in line.split("\t")]
                if len(parts) < 4 or parts[0] != series_id:
                    continue
                try:
                    year = int(parts[1])
                except ValueError:
                    continue
                period = parts[2]
                if not (HISTORY_START_YEAR <= year <= end_year and re.fullmatch(r"M\d{2}", period)):
                    continue
                rows.append({
                    "series_id": series_id,
                    "year": str(year),
                    "period": period,
                    "employment_thousands": parts[3],
                })
        with cache.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=["series_id", "year", "period", "employment_thousands"])
            w.writeheader()
            w.writerows(rows)

    by_month: dict[str, float] = {}
    for row in rows:
        month = int(row["period"][1:])
        key = f'{int(row["year"]):04d}-{month:02d}'
        by_month[key] = float(row["employment_thousands"]) * 1000.0

    months = [f"{year}-{month:02d}" for year in range(HISTORY_START_YEAR, end_year + 1) for month in range(1, 13)]
    missing = [m for m in months if m not in by_month]
    if missing:
        raise RuntimeError(f"Missing CES employment months for {series_id}: {missing[:10]}")
    values = np.asarray([by_month[m] for m in months], dtype=np.float64)
    return months, values, cache


def _xlsx_sheet_rows(path: Path, sheet_name: str) -> list[dict[int, str]]:
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as z:
        wb = ET.fromstring(z.read("xl/workbook.xml"))
        rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        ridmap = {x.attrib["Id"]: x.attrib["Target"] for x in rels}

        shared: list[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            ss = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in ss.findall("m:si", ns):
                shared.append("".join(t.text or "" for t in si.iterfind(".//m:t", ns)))

        target = None
        for sheet in wb.findall("m:sheets/m:sheet", ns):
            if sheet.attrib["name"] == sheet_name:
                rid = sheet.attrib[
                    "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
                ]
                target = ridmap[rid]
                break
        if target is None:
            raise KeyError(f"Sheet {sheet_name!r} not found in {path.name}")

        sheet_path = target.lstrip("/") if target.startswith("/") else str(PurePosixPath("xl") / target)
        root = ET.fromstring(z.read(sheet_path))

        def col_num(ref: str) -> int:
            m = re.match(r"([A-Z]+)", ref)
            if not m:
                raise ValueError(ref)
            n = 0
            for ch in m.group(1):
                n = n * 26 + ord(ch) - 64
            return n

        rows: list[dict[int, str]] = []
        for row in root.findall(".//m:sheetData/m:row", ns):
            out: dict[int, str] = {}
            for cell in row.findall("m:c", ns):
                ref = cell.attrib.get("r", "")
                typ = cell.attrib.get("t")
                value = ""
                v = cell.find("m:v", ns)
                inline = cell.find("m:is", ns)
                if typ == "s" and v is not None:
                    value = shared[int(v.text)]
                elif typ == "inlineStr" and inline is not None:
                    value = "".join(t.text or "" for t in inline.iterfind(".//m:t", ns))
                elif v is not None:
                    value = v.text or ""
                out[col_num(ref)] = value
            rows.append(out)
        return rows


def read_bea_quarterly(path: Path, sheet_name: str) -> tuple[list[str], dict[str, np.ndarray]]:
    rows = _xlsx_sheet_rows(path, sheet_name)
    header_idx = next(i for i, row in enumerate(rows) if str(row.get(1, "")).strip() == "Line")
    header = rows[header_idx]
    cols = [c for c in sorted(header) if c >= 4 and re.fullmatch(r"\d{4}Q[1-4]", str(header[c]).strip())]
    periods = [str(header[c]).strip() for c in cols]

    table: dict[str, np.ndarray] = {}
    for row in rows[header_idx + 1 :]:
        name = str(row.get(2, "")).strip()
        if not name or name.startswith(("1.", "2.", "3.", "4.", "Addenda", "Note.")):
            continue
        vals: list[float] = []
        for c in cols:
            raw = str(row.get(c, "")).replace(",", "").strip()
            try:
                vals.append(float(raw))
            except ValueError:
                vals = []
                break
        if len(vals) == len(cols):
            table[name] = np.asarray(vals, dtype=np.float64)
    if not table:
        raise RuntimeError(f"No quarterly observations parsed from {path.name}:{sheet_name}")
    return periods, table


def read_bea_annual(path: Path, sheet_name: str) -> tuple[list[int], dict[str, np.ndarray]]:
    rows = _xlsx_sheet_rows(path, sheet_name)
    header_idx = next(i for i, row in enumerate(rows) if str(row.get(1, "")).strip() == "Line")
    header = rows[header_idx]
    cols = [c for c in sorted(header) if c >= 4 and re.fullmatch(r"\d{4}", str(header[c]).strip())]
    years = [int(str(header[c]).strip()) for c in cols]

    table: dict[str, np.ndarray] = {}
    for row in rows[header_idx + 1 :]:
        name = str(row.get(2, "")).strip()
        if not name or name.startswith(("1.", "2.", "3.", "4.", "Addenda", "Note.")):
            continue
        vals: list[float] = []
        for c in cols:
            raw = str(row.get(c, "")).replace(",", "").strip()
            try:
                vals.append(float(raw))
            except ValueError:
                vals = []
                break
        if len(vals) == len(cols):
            table[name] = np.asarray(vals, dtype=np.float64)
    if not table:
        raise RuntimeError(f"No annual observations parsed from {path.name}:{sheet_name}")
    return years, table


def read_bea_annual_gos(path: Path) -> tuple[list[int], dict[str, np.ndarray]]:
    rows = _xlsx_sheet_rows(path, "TVA113-A")
    header_idx = next(i for i, row in enumerate(rows) if str(row.get(1, "")).strip() == "Line")
    header = rows[header_idx]
    cols = [c for c in sorted(header) if c >= 4 and re.fullmatch(r"\d{4}", str(header[c]).strip())]
    years = [int(str(header[c]).strip()) for c in cols]

    components = {
        "Compensation of employees",
        "Taxes on production and imports less subsidies",
        "Gross operating surplus",
    }
    table: dict[str, np.ndarray] = {}
    current_industry: str | None = None
    for row in rows[header_idx + 1 :]:
        name = str(row.get(2, "")).strip()
        if not name:
            continue
        if name not in components and not name.startswith(("1.", "2.", "3.", "Addenda", "Note.")):
            current_industry = name
            continue
        if name != "Gross operating surplus" or current_industry is None:
            continue
        vals: list[float] = []
        for c in cols:
            raw = str(row.get(c, "")).replace(",", "").strip()
            try:
                vals.append(float(raw))
            except ValueError:
                vals = []
                break
        if len(vals) == len(cols):
            table[current_industry] = np.asarray(vals, dtype=np.float64)
    if not table:
        raise RuntimeError("No annual GOS observations parsed from TVA113-A")
    return years, table


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


def quarter_ref(text: str) -> str | None:
    m = re.search(
        r"(First|1st|Second|2nd|Third|3rd|Fourth|4th)\s+Quarter(?:\s+and\s+(?:Annual|Year))?\s*,?\s*(\d{4})",
        text,
        re.I,
    )
    if not m:
        return None
    return f"{int(m.group(2))}Q{QWORDS[m.group(1).lower()]}"


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


def collect_bea_release_metadata(session: requests.Session) -> tuple[dict[str, pd.Timestamp], dict[int, pd.Timestamp], list[dict[str, str]]]:
    archive_rows: list[tuple[str, pd.Timestamp, str]] = []
    for page in range(20):
        url = f"https://www.bea.gov/news/archive?created_1=All&field_related_product_target_id=456&page={page}&title="
        soup = BeautifulSoup(session.get(url, timeout=30).text, "html.parser")
        table = soup.find("table")
        if table is None:
            continue
        for tr in table.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all("td")]
            if len(cells) < 2:
                continue
            try:
                release_date = pd.Timestamp(pd.to_datetime(cells[1])).normalize()
            except Exception:
                continue
            archive_rows.append((cells[0], release_date, url))

    q_release: dict[str, pd.Timestamp] = {}
    rows_out: list[dict[str, str]] = []
    for title, release_date, url in archive_rows:
        if "industry" not in title.casefold():
            continue
        ref = quarter_ref(title)
        if ref:
            q_release[ref] = min(q_release.get(ref, release_date), release_date)

    # 71-industry underlying detail became available with history to 2012Q1 in
    # Nov-2015, and the 2018 comprehensive update extended that detail to 2005Q1.
    for p in pd.period_range("2012Q1", "2015Q2", freq="Q"):
        q_release[str(p)] = pd.Timestamp("2015-11-05")
    for p in pd.period_range("2005Q1", "2011Q4", freq="Q"):
        q_release[str(p)] = pd.Timestamp("2018-11-01")

    # Annual industry accounts existed before the quarterly product. For the
    # detailed annual components used by PRISM, accept only releases that are
    # explicitly annual/advance/revised annual accounts; do not mistake an
    # ordinary quarterly release that happens to mention the year for annual data.
    annual_release: dict[int, pd.Timestamp] = {}
    for year in range(1997, 2025):
        candidates: list[pd.Timestamp] = []
        for title, release_date, _ in archive_rows:
            lower = title.casefold()
            if "industry" not in lower:
                continue
            if release_date <= pd.Timestamp(year=year, month=12, day=31):
                continue
            if release_date > pd.Timestamp(year=year + 2, month=12, day=31):
                continue
            explicit_year = str(year) in title
            pre_quarterly_annual = explicit_year and (
                "annual" in lower
                or "advance gross domestic product by industry" in lower
                or "revised statistics of gross domestic product by industry" in lower
                or "revised estimates of gross domestic product by industry" in lower
                or "revised" in lower
                or f"by industry for {year}" in lower
                or re.search(rf"gross domestic product by industry[:,]?\s*{year}\b", lower) is not None
            )
            annual_update = (
                release_date.year == year + 1
                and ("annual update" in lower or "comprehensive update" in lower)
            )
            if pre_quarterly_annual or annual_update:
                candidates.append(release_date)
        if candidates:
            annual_release[year] = min(candidates)

    # For the modern quarterly system, detailed annual component tables are
    # updated with the following year's Q2 industry release / annual update.
    for year in range(2015, 2025):
        q2 = f"{year + 1}Q2"
        if q2 in q_release:
            annual_release[year] = q_release[q2]

    for q, release_date in sorted(q_release.items()):
        rows_out.append({
            "family": "BEA_QUARTERLY_71",
            "reference_period": q,
            "release_date": release_date.date().isoformat(),
            "source_url": "https://www.bea.gov/news/",
        })
    for year, release_date in sorted(annual_release.items()):
        rows_out.append({
            "family": "BEA_ANNUAL_INDUSTRY",
            "reference_period": str(year),
            "release_date": release_date.date().isoformat(),
            "source_url": "https://www.bea.gov/news/archive",
        })
    return q_release, annual_release, rows_out


def collect_bls_target_release_metadata(
    session: requests.Session,
) -> tuple[
    dict[str, pd.Timestamp],
    dict[str, pd.Timestamp],
    dict[str, pd.Timestamp],
    list[dict[str, str]],
]:
    rows_out: list[dict[str, str]] = []

    # QCEW: use the historical release archive. The full-data release is the
    # relevant availability date for the detailed NAICS observations used here.
    qcew_url = "https://www.bls.gov/bls/news-release/cewqtr.htm"
    soup = BeautifulSoup(session.get(qcew_url, timeout=30).text, "html.parser")
    qcew_release: dict[str, pd.Timestamp] = {}
    for a in soup.find_all("a"):
        href = str(a.get("href", ""))
        if "/news.release/history/cewqtr_" not in href or not href.lower().endswith(".txt"):
            continue
        release_date = release_date_from_archive_href(href)
        if release_date is None:
            continue
        url = urljoin(qcew_url, href)
        text = session.get(url, timeout=30).text[:12000]
        m = re.search(
            r"COUNTY EMPLOYMENT AND WAGES\s*[:\-]?\s*"
            r"(FIRST|SECOND|THIRD|FOURTH|1ST|2ND|3RD|4TH)\s+QUARTER\s+(\d{4})",
            text,
            re.I,
        )
        if not m:
            continue
        ref = f"{int(m.group(2))}Q{QWORDS[m.group(1).lower()]}"
        qcew_release[ref] = release_date

    # Yearly BLS schedules cover the middle historical span more completely.
    for year in range(HISTORY_START_YEAR, 2026):
        url = f"https://www.bls.gov/schedule/{year}/home.htm"
        ysoup = BeautifulSoup(session.get(url, timeout=30).text, "html.parser")
        entries: list[tuple[str, str]] = []
        for table in ysoup.find_all("table"):
            for tr in table.find_all("tr"):
                cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
                if len(cells) >= 3:
                    entries.append((cells[0], cells[2]))
        pre = ysoup.find("pre")
        if pre:
            for line in pre.get_text().splitlines():
                if "County Employment and Wages" not in line:
                    continue
                parsed = parse_pre_schedule_entry(line, year)
                if parsed is not None:
                    desc, release_date = parsed
                    entries.append((release_date.date().isoformat(), desc))
        for date_text, desc in entries:
            if "county employment and wages" not in desc.casefold():
                continue
            ref = quarter_ref(desc)
            release_date = parse_release_date(date_text, year)
            if ref and release_date is not None and release_date.year < int(ref[:4]):
                release_date = release_date + pd.DateOffset(years=1)
            if ref and release_date is not None:
                qcew_release[ref] = max(qcew_release.get(ref, release_date), release_date)

    # From 2017Q4 onward BLS publishes a separate full-data date; use that
    # rather than the earlier news-release date when available.
    calendar_url = "https://www.bls.gov/cew/release-calendar.htm"
    csoup = BeautifulSoup(session.get(calendar_url, timeout=30).text, "html.parser")
    current_year: int | None = None
    for elem in csoup.find_all(["h2", "table"]):
        if elem.name == "h2":
            ym = re.search(r"(20\d{2})", elem.get_text(" ", strip=True))
            current_year = int(ym.group(1)) if ym else current_year
            continue
        if current_year is None:
            continue
        for tr in elem.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
            if not cells or "quarter" not in cells[0].casefold():
                continue
            qm = re.search(r"([1-4])(?:st|nd|rd|th)\s+Quarter", cells[0], re.I)
            if not qm:
                continue
            ref = f"{current_year}Q{qm.group(1)}"
            date_text = cells[3] if len(cells) >= 4 and re.search(r"\d{4}", cells[3]) else cells[1]
            release_date = parse_release_date(date_text, current_year + 1)
            if release_date is not None:
                qcew_release[ref] = release_date

    # CES corrections: Employment Situation release dates for each reference month.
    monthly_ces: dict[str, pd.Timestamp] = {}
    for year in range(HISTORY_START_YEAR, 2026):
        url = f"https://www.bls.gov/schedule/{year}/home.htm"
        soup = BeautifulSoup(session.get(url, timeout=30).text, "html.parser")
        entries: list[tuple[str, str]] = []
        for table in soup.find_all("table"):
            for tr in table.find_all("tr"):
                cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
                if len(cells) >= 3:
                    entries.append((cells[0], cells[2]))
        pre = soup.find("pre")
        if pre:
            for line in pre.get_text().splitlines():
                if "Employment Situation" not in line:
                    continue
                parsed = parse_pre_schedule_entry(line, year)
                if parsed is not None:
                    desc, release_date = parsed
                    entries.append((release_date.date().isoformat(), desc))
        for date_text, desc in entries:
            if "employment situation" not in desc.casefold():
                continue
            m = re.search(
                r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})",
                desc,
                re.I,
            )
            if not m:
                continue
            ref = f"{int(m.group(2)):04d}-{MONTHS[m.group(1).lower()]:02d}"
            release_date = parse_release_date(date_text, year)
            if release_date is not None and release_date < pd.Timestamp(f"{ref}-01"):
                release_date = release_date + pd.DateOffset(years=1)
            if release_date is not None:
                monthly_ces[ref] = release_date

    # Fill any old schedule gaps from the official Employment Situation archive.
    required_months = set(
        pd.period_range(f"{HISTORY_START_YEAR}-01", "2024-12", freq="M").astype(str)
    )
    missing_months = required_months - set(monthly_ces)
    if missing_months:
        archive_url = "https://www.bls.gov/bls/news-release/empsit.htm"
        asoup = BeautifulSoup(session.get(archive_url, timeout=30).text, "html.parser")
        for a in asoup.find_all("a"):
            href = str(a.get("href", ""))
            if "/news.release/history/empsit_" not in href or not href.lower().endswith(".txt"):
                continue
            release_date = release_date_from_archive_href(href)
            if release_date is None:
                continue
            estimated_ref = str(release_date.to_period("M") - 1)
            if estimated_ref not in missing_months:
                continue
            url = urljoin(archive_url, href)
            text = session.get(url, timeout=30).text[:12000]
            match = re.search(
                r"EMPLOYMENT SITUATION\s*:\s*"
                r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})",
                text,
                re.I,
            )
            if not match:
                continue
            ref = f"{int(match.group(2)):04d}-{MONTHS[match.group(1).lower()]:02d}"
            if ref not in missing_months:
                continue
            monthly_ces[ref] = release_date
            missing_months.remove(ref)
            if not missing_months:
                break

    ces_release: dict[str, pd.Timestamp] = {}
    for q in pd.period_range(f"{HISTORY_START_YEAR}Q1", "2024Q4", freq="Q"):
        months = [str(m) for m in pd.period_range(q.start_time, q.end_time, freq="M")]
        if all(m in monthly_ces for m in months):
            ces_release[str(q)] = max(monthly_ces[m] for m in months)

    for q, release_date in sorted(qcew_release.items()):
        rows_out.append({
            "family": "QCEW_QUARTERLY",
            "reference_period": q,
            "release_date": release_date.date().isoformat(),
            "source_url": qcew_url,
        })
    for q, release_date in sorted(ces_release.items()):
        rows_out.append({
            "family": "CES_QUARTERLY",
            "reference_period": q,
            "release_date": release_date.date().isoformat(),
            "source_url": "https://www.bls.gov/schedule/",
        })
    for month, release_date in sorted(monthly_ces.items()):
        rows_out.append({
            "family": "CES_MONTHLY",
            "reference_period": month,
            "release_date": release_date.date().isoformat(),
            "source_url": "https://www.bls.gov/schedule/",
        })
    return qcew_release, ces_release, monthly_ces, rows_out


def lookup(table: dict[str, np.ndarray], name: str) -> tuple[str, np.ndarray]:
    target = name.casefold()
    for key, values in table.items():
        if key.casefold() == target:
            return key, values
    raise KeyError(name)


def node_bea_series(
    nodes: list[dict[str, str]], table: dict[str, np.ndarray]
) -> tuple[np.ndarray, list[dict[str, str]]]:
    cols: list[np.ndarray] = []
    mapping: list[dict[str, str]] = []
    for node in nodes:
        key = node["key"]
        if key == "hospitals_nursing":
            k1, v1 = lookup(table, "Hospitals")
            k2, v2 = lookup(table, "Nursing and residential care facilities")
            cols.append(v1 + v2)
            mapping.append({"node_key": key, "bea_rows": f"{k1} + {k2}", "mapping": "COMPOSITE_SUM"})
        else:
            requested = BEA_ALIAS.get(key, node["name"])
            actual, values = lookup(table, requested)
            cols.append(values)
            mapping.append({"node_key": key, "bea_rows": actual, "mapping": "DIRECT"})
    return np.stack(cols, axis=1), mapping


def qcew_codes(node: dict[str, str]) -> list[str]:
    # QCEW exposes combined retail as NAICS 44-45 rather than two separate series.
    if node["key"] == "retail":
        return ["44-45"]
    return [x.strip() for x in node["naics_codes"].split(",") if x.strip()]


def fetch_qcew(
    session: requests.Session,
    nodes: list[dict[str, str]],
    end_year: int,
    refresh: bool,
) -> tuple[list[str], np.ndarray, dict[str, dict[str, float]], list[dict[str, str]]]:
    cache = RAW_BLS_DIR / f"qcew_monthly_employment_{HISTORY_START_YEAR}_{end_year}.json"
    all_codes = sorted({code for node in nodes for code in qcew_codes(node)})

    if cache.exists() and not refresh:
        raw: dict[str, dict[str, float]] = json.loads(cache.read_text(encoding="utf-8"))
    else:
        raw = {code: {} for code in all_codes}
        # BLS public API accepts at most ten years per request; use two 10-year blocks here.
        ranges: list[tuple[int, int]] = []
        y = HISTORY_START_YEAR
        while y <= end_year:
            y1 = min(y + 9, end_year)
            ranges.append((y, y1))
            y = y1 + 1

        for start, end in ranges:
            for offset in range(0, len(all_codes), 20):
                codes = all_codes[offset : offset + 20]
                series_ids = ["ENUUS000105" + code for code in codes]
                payload = {"seriesid": series_ids, "startyear": str(start), "endyear": str(end)}

                obj = None
                for attempt in range(4):
                    try:
                        response = session.post(BLS_API_URL, json=payload, timeout=90)
                        response.raise_for_status()
                        obj = response.json()
                        if obj.get("status") != "REQUEST_SUCCEEDED":
                            raise RuntimeError(obj)
                        break
                    except Exception:
                        if attempt == 3:
                            raise
                        time.sleep(2 ** attempt)
                assert obj is not None

                by_id = {s["seriesID"]: s.get("data", []) for s in obj["Results"]["series"]}
                for code, series_id in zip(codes, series_ids):
                    rows = by_id.get(series_id, [])
                    if not rows:
                        raise RuntimeError(f"BLS QCEW series returned no data: {series_id} (NAICS {code})")
                    for row in rows:
                        period = row["period"]
                        if not re.fullmatch(r"M\d{2}", period):
                            continue
                        month = f'{row["year"]}-{int(period[1:]):02d}'
                        raw[code][month] = float(row["value"])
                time.sleep(0.15)

        cache.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")

    months = [f"{year}-{month:02d}" for year in range(HISTORY_START_YEAR, end_year + 1) for month in range(1, 13)]
    panel = np.empty((len(months), len(nodes)), dtype=np.float64)
    mappings: list[dict[str, str]] = []
    for j, node in enumerate(nodes):
        codes = qcew_codes(node)
        mappings.append({
            "node_key": node["key"],
            "qcew_naics": ";".join(codes),
            "aggregation": "SUM" if len(codes) > 1 else "DIRECT",
        })
        for i, month in enumerate(months):
            values = [raw.get(code, {}).get(month) for code in codes]
            if any(v is None for v in values):
                raise RuntimeError(f"Missing QCEW employment: node={node['key']} codes={codes} month={month}")
            panel[i, j] = float(sum(v for v in values if v is not None))

    return months, panel, raw, mappings


def write_wide_csv(path: Path, periods: list[str], nodes: list[dict[str, str]], values: np.ndarray, period_name: str) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([period_name] + [n["key"] for n in nodes])
        for period, row in zip(periods, values):
            writer.writerow([period] + [f"{float(x):.12g}" for x in row])


def write_bea_quarterly_long(
    path: Path,
    quarters: list[str],
    nodes: list[dict[str, str]],
    current_va: np.ndarray,
    real_va: np.ndarray,
    current_go: np.ndarray,
    real_go: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        fields = ["quarter", "node_key", "node_name", "current_va", "real_va", "current_go", "real_go"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for t, quarter in enumerate(quarters):
            for j, node in enumerate(nodes):
                w.writerow({
                    "quarter": quarter,
                    "node_key": node["key"],
                    "node_name": node["name"],
                    "current_va": f"{current_va[t, j]:.12g}",
                    "real_va": f"{real_va[t, j]:.12g}",
                    "current_go": f"{current_go[t, j]:.12g}",
                    "real_go": f"{real_go[t, j]:.12g}",
                })


def write_annual_gos_long(path: Path, years: list[int], nodes: list[dict[str, str]], gos: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["year", "node_key", "node_name", "gross_operating_surplus"])
        for t, year in enumerate(years):
            for j, node in enumerate(nodes):
                w.writerow([year, node["key"], node["name"], f"{gos[t, j]:.12g}"])


def write_employment_long(path: Path, periods: list[str], nodes: list[dict[str, str]], values: np.ndarray, period_name: str) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow([period_name, "node_key", "node_name", "employment"])
        for t, period in enumerate(periods):
            for j, node in enumerate(nodes):
                w.writerow([period, node["key"], node["name"], f"{values[t, j]:.12g}"])


def write_mapping(path: Path, bea_mapping: list[dict[str, str]], qcew_mapping: list[dict[str, str]]) -> None:
    qcew = {r["node_key"]: r for r in qcew_mapping}
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        fields = ["node_key", "bea_rows", "bea_mapping", "qcew_naics", "qcew_aggregation"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in bea_mapping:
            q = qcew[row["node_key"]]
            w.writerow({
                "node_key": row["node_key"],
                "bea_rows": row["bea_rows"],
                "bea_mapping": row["mapping"],
                "qcew_naics": q["qcew_naics"],
                "qcew_aggregation": q["aggregation"],
            })


def collect(refresh: bool) -> None:
    ensure_dirs()
    nodes = load_nodes()
    session = requests.Session()
    session.headers.update({"User-Agent": "PRISM academic research haseung.ryu.ai@gmail.com"})

    print("[1/5] Downloading BEA workbooks...")
    value_added_path = RAW_BEA_DIR / "ValueAdded.xlsx"
    gross_output_path = RAW_BEA_DIR / "GrossOutput.xlsx"
    manifest = [
        {"source": "BEA", "dataset": "GDP by Industry - Value Added", **download_file(session, BEA_VALUE_ADDED_URL, value_added_path, refresh)},
        {"source": "BEA", "dataset": "GDP by Industry - Gross Output", **download_file(session, BEA_GROSS_OUTPUT_URL, gross_output_path, refresh)},
    ]

    print("[2/5] Parsing BEA target inputs and validating 56-node mapping...")
    q_cva, t_cva = read_bea_quarterly(value_added_path, "TVA105-Q")
    q_rva, t_rva = read_bea_quarterly(value_added_path, "TVA106-Q")
    q_cgo, t_cgo = read_bea_quarterly(gross_output_path, "TGO105-Q")
    q_rgo, t_rgo = read_bea_quarterly(gross_output_path, "TGO106-Q")
    if not (q_cva == q_rva == q_cgo == q_rgo):
        raise RuntimeError("BEA quarterly period columns do not match across required tables")

    current_va, mapping_cva = node_bea_series(nodes, t_cva)
    real_va, mapping_rva = node_bea_series(nodes, t_rva)
    current_go, mapping_cgo = node_bea_series(nodes, t_cgo)
    real_go, mapping_rgo = node_bea_series(nodes, t_rgo)
    if not (mapping_cva == mapping_rva == mapping_cgo == mapping_rgo):
        raise RuntimeError("BEA node mapping differs across required quarterly tables")

    gos_years_all, t_gos = read_bea_annual_gos(value_added_path)
    annual_gos_all, mapping_gos = node_bea_series(nodes, t_gos)
    if mapping_gos != mapping_cva:
        raise RuntimeError("BEA node mapping differs between quarterly tables and annual GOS")

    y_cva, a_cva = read_bea_annual(value_added_path, "TVA105-A")
    y_rva, a_rva = read_bea_annual(value_added_path, "TVA106-A")
    y_cgo, a_cgo = read_bea_annual(gross_output_path, "TGO105-A")
    y_rgo, a_rgo = read_bea_annual(gross_output_path, "TGO106-A")
    if not (y_cva == y_rva == y_cgo == y_rgo):
        raise RuntimeError("BEA annual VA/GO period columns do not match")
    missing_annual = sorted(set(gos_years_all) - set(y_cva))
    if missing_annual:
        raise RuntimeError(f"Annual VA/GO missing GOS years: {missing_annual}")
    annual_current_va_all, mapping_acva = node_bea_series(nodes, a_cva)
    annual_real_va_all, mapping_arva = node_bea_series(nodes, a_rva)
    annual_current_go_all, mapping_acgo = node_bea_series(nodes, a_cgo)
    annual_real_go_all, mapping_argo = node_bea_series(nodes, a_rgo)
    if not (mapping_acva == mapping_arva == mapping_acgo == mapping_argo == mapping_gos):
        raise RuntimeError("BEA annual node mapping differs across target-history tables")

    latest_gos_year = max(y for y in gos_years_all if y >= TARGET_START_YEAR)
    gos_keep = [i for i, y in enumerate(gos_years_all) if TARGET_START_YEAR <= y <= latest_gos_year]
    gos_years = [gos_years_all[i] for i in gos_keep]
    annual_gos = annual_gos_all[gos_keep]

    bea_history_years = [y for y in gos_years_all if y <= latest_gos_year]
    history_years = [y for y in gos_years_all if HISTORY_START_YEAR <= y <= latest_gos_year]
    annual_index = {y: i for i, y in enumerate(y_cva)}
    gos_index = {y: i for i, y in enumerate(gos_years_all)}
    history_annual_current_va = np.stack([annual_current_va_all[annual_index[y]] for y in bea_history_years])
    history_annual_real_va = np.stack([annual_real_va_all[annual_index[y]] for y in bea_history_years])
    history_annual_current_go = np.stack([annual_current_go_all[annual_index[y]] for y in bea_history_years])
    history_annual_real_go = np.stack([annual_real_go_all[annual_index[y]] for y in bea_history_years])
    history_annual_gos = np.stack([annual_gos_all[gos_index[y]] for y in bea_history_years])

    target_quarter_end = f"{latest_gos_year}Q4"
    q_keep = [i for i, q in enumerate(q_cva) if f"{TARGET_START_YEAR}Q1" <= q <= target_quarter_end]
    target_quarters = [q_cva[i] for i in q_keep]
    if not target_quarters or target_quarters[0] != f"{TARGET_START_YEAR}Q1" or target_quarters[-1] != target_quarter_end:
        raise RuntimeError(f"BEA quarterly coverage is incomplete for {TARGET_START_YEAR}Q1..{target_quarter_end}")

    print(f"[3/5] Downloading BLS employment ({HISTORY_START_YEAR}-{latest_gos_year})...")
    months, employment_monthly, _, qcew_mapping = fetch_qcew(session, nodes, latest_gos_year, refresh)
    target_month_mask = np.asarray([m >= f"{TARGET_START_YEAR}-01" for m in months])
    target_months = [m for m, keep in zip(months, target_month_mask) if keep]
    employment_monthly_target = employment_monthly[target_month_mask]
    employment_quarterly = employment_monthly_target.reshape(len(target_quarters), 3, EXPECTED_NODE_COUNT).mean(axis=1)

    history_year_count = len(history_years)
    if len(months) != history_year_count * 12:
        raise RuntimeError("History employment month grid does not align with annual BEA history years")
    history_annual_employment = employment_monthly.reshape(
        history_year_count, 12, EXPECTED_NODE_COUNT
    ).mean(axis=1)

    # Prefer exact CES employment for industries with an official publication-
    # vintage series. This keeps the ex-post target denominator aligned with the
    # point-in-time denominator while avoiding QCEW's multi-quarter finalization
    # delay for those nodes. Nodes without an exact CES vintage remain on QCEW.
    ces_exact_mapping = build_ces_exact_vintage_mapping(
        session, nodes, RAW_BLS_DIR, refresh
    )
    ces_exact_vintages = load_exact_ces_vintages(
        session,
        ces_exact_mapping,
        RAW_BLS_DIR,
        False,
        HISTORY_START_YEAR - 1,
        latest_gos_year,
    )
    ces_exact_final_monthly = current_final_ces_monthly_panel(
        ces_exact_vintages, months
    )
    ces_exact_mapping = ces_exact_mapping.copy()
    ces_exact_mapping["used_for_target_employment"] = ces_exact_mapping["node_key"].isin(
        ces_exact_final_monthly
    )
    ces_exact_mapping.to_csv(
        RESULTS_DIR / "ces_vintage_node_mapping.csv", index=False, encoding="utf-8-sig"
    )

    # QCEW structurally excludes railroad workers covered by the railroad
    # unemployment insurance system. Correct only the rail_transport target
    # denominator with official CES Rail Transportation employment (NAICS 482).
    rail_months, rail_monthly, rail_cache = fetch_ces_employment_series(
        session, RAIL_CES_SERIES, "rail_transport_ces_nsa", latest_gos_year, refresh
    )
    if rail_months != months:
        raise RuntimeError("CES rail/QCEW month grids differ")
    rail_monthly_target = rail_monthly[target_month_mask]
    rail_quarterly = rail_monthly_target.reshape(len(target_quarters), 3).mean(axis=1)
    rail_annual = rail_monthly.reshape(history_year_count, 12).mean(axis=1)
    rail_idx = next(i for i, n in enumerate(nodes) if n["key"] == "rail_transport")

    funds_months, funds_monthly, funds_cache = fetch_ces_employment_series(
        session, FUNDS_TRUSTS_CES_SERIES, "funds_trusts_ces_nsa", latest_gos_year, refresh
    )
    if funds_months != months:
        raise RuntimeError("CES funds/QCEW month grids differ")
    funds_monthly_target = funds_monthly[target_month_mask]
    funds_quarterly = funds_monthly_target.reshape(len(target_quarters), 3).mean(axis=1)
    funds_annual = funds_monthly.reshape(history_year_count, 12).mean(axis=1)
    funds_idx = next(i for i, n in enumerate(nodes) if n["key"] == "funds_trusts")

    target_employment_quarterly = employment_quarterly.copy()
    history_annual_target_employment = history_annual_employment.copy()
    for j, node in enumerate(nodes):
        exact_monthly = ces_exact_final_monthly.get(node["key"])
        if exact_monthly is None:
            continue
        exact_target = exact_monthly[target_month_mask]
        target_employment_quarterly[:, j] = exact_target.reshape(
            len(target_quarters), 3
        ).mean(axis=1)
        history_annual_target_employment[:, j] = exact_monthly.reshape(
            history_year_count, 12
        ).mean(axis=1)

    # Funds/trusts is intentionally defined with a continuous CES 523+525 proxy
    # and a matching BEA numerator, so it overrides the generic fallback.
    history_annual_target_employment[:, funds_idx] = funds_annual
    target_employment_quarterly[:, funds_idx] = funds_quarterly

    print("[3b/5] Collecting publication metadata with target sources...")
    bea_quarterly_release, bea_annual_release, bea_release_rows = collect_bea_release_metadata(session)
    qcew_release, ces_release, monthly_ces_release, bls_release_rows = collect_bls_target_release_metadata(session)
    required_qcew = {str(q) for q in pd.period_range(f"{HISTORY_START_YEAR}Q1", target_quarter_end, freq="Q")}
    missing_qcew_release = sorted(required_qcew - set(qcew_release))
    if missing_qcew_release:
        raise RuntimeError(f"QCEW release metadata gaps: {missing_qcew_release[:10]}")
    if HISTORY_START_YEAR not in bea_annual_release:
        raise RuntimeError(f"Missing BEA annual release metadata for {HISTORY_START_YEAR}")
    if f"{HISTORY_START_YEAR}Q4" not in ces_release:
        raise RuntimeError(f"Missing CES release metadata for {HISTORY_START_YEAR}Q4")
    release_rows = bea_release_rows + bls_release_rows
    with RELEASE_METADATA_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["family", "reference_period", "release_date", "source_url"],
        )
        w.writeheader()
        w.writerows(release_rows)

    print("[3c/5] Building revision-safe point-in-time target history...")
    pit_history_summary = build_point_in_time_history(
        session=session,
        nodes=nodes,
        bea_alias=BEA_ALIAS,
        model_start="2006Q1",
        model_end=target_quarter_end,
        qcew_months=months,
        qcew_monthly=employment_monthly,
        qcew_release=qcew_release,
        ces_monthly_release=monthly_ces_release,
        raw_bea_dir=RAW_BEA_DIR,
        raw_bls_dir=RAW_BLS_DIR,
        output_csv=PIT_HISTORY_RAW_CSV,
        audit_csv=PIT_HISTORY_AUDIT_CSV,
        refresh=refresh,
        ces_exact_mapping=ces_exact_mapping,
        ces_exact_vintages=ces_exact_vintages,
    )

    print("[4/5] Writing node-aligned target input tables...")
    current_va_target = current_va[q_keep]
    real_va_target = real_va[q_keep]
    current_go_target = current_go[q_keep]
    real_go_target = real_go[q_keep]

    write_bea_quarterly_long(
        PROCESSED_DIR / "bea_quarterly_56_long.csv",
        target_quarters,
        nodes,
        current_va_target,
        real_va_target,
        current_go_target,
        real_go_target,
    )
    write_annual_gos_long(PROCESSED_DIR / "bea_annual_gos_56_long.csv", gos_years, nodes, annual_gos)
    write_employment_long(PROCESSED_DIR / "qcew_monthly_employment_56_long.csv", months, nodes, employment_monthly, "month")
    write_employment_long(PROCESSED_DIR / "qcew_quarterly_employment_56_long.csv", target_quarters, nodes, employment_quarterly, "quarter")
    write_employment_long(PROCESSED_DIR / "target_quarterly_employment_56_long.csv", target_quarters, nodes, target_employment_quarterly, "quarter")

    # Wide forms are convenient for direct numerical/model ingestion.
    write_wide_csv(PROCESSED_DIR / "current_value_added_56.csv", target_quarters, nodes, current_va_target, "quarter")
    write_wide_csv(PROCESSED_DIR / "real_value_added_56.csv", target_quarters, nodes, real_va_target, "quarter")
    write_wide_csv(PROCESSED_DIR / "current_gross_output_56.csv", target_quarters, nodes, current_go_target, "quarter")
    write_wide_csv(PROCESSED_DIR / "real_gross_output_56.csv", target_quarters, nodes, real_go_target, "quarter")
    write_wide_csv(PROCESSED_DIR / "qcew_quarterly_employment_56.csv", target_quarters, nodes, employment_quarterly, "quarter")
    write_wide_csv(PROCESSED_DIR / "target_quarterly_employment_56.csv", target_quarters, nodes, target_employment_quarterly, "quarter")
    write_wide_csv(PROCESSED_DIR / "annual_gos_56.csv", [str(y) for y in gos_years], nodes, annual_gos, "year")
    write_wide_csv(PROCESSED_DIR / "history_annual_current_value_added_56.csv", [str(y) for y in bea_history_years], nodes, history_annual_current_va, "year")
    write_wide_csv(PROCESSED_DIR / "history_annual_real_value_added_56.csv", [str(y) for y in bea_history_years], nodes, history_annual_real_va, "year")
    write_wide_csv(PROCESSED_DIR / "history_annual_current_gross_output_56.csv", [str(y) for y in bea_history_years], nodes, history_annual_current_go, "year")
    write_wide_csv(PROCESSED_DIR / "history_annual_real_gross_output_56.csv", [str(y) for y in bea_history_years], nodes, history_annual_real_go, "year")
    write_wide_csv(PROCESSED_DIR / "history_annual_gos_56.csv", [str(y) for y in bea_history_years], nodes, history_annual_gos, "year")
    write_wide_csv(PROCESSED_DIR / "history_annual_employment_56.csv", [str(y) for y in history_years], nodes, history_annual_target_employment, "year")
    write_mapping(RESULTS_DIR / "node_source_mapping.csv", mapping_cva, qcew_mapping)

    with (RESULTS_DIR / "rail_transport_employment_correction.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["quarter", "qcew_employment", "ces_rail_employment", "ces_to_qcew_ratio"])
        for q, qcew_v, ces_v in zip(target_quarters, employment_quarterly[:, rail_idx], rail_quarterly):
            w.writerow([q, f"{qcew_v:.12g}", f"{ces_v:.12g}", f"{ces_v / qcew_v:.12g}"])

    with (RESULTS_DIR / "funds_trusts_employment_correction.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["quarter", "qcew_employment", "ces_proxy_employment", "ces_to_qcew_ratio"])
        for q, qcew_v, ces_v in zip(target_quarters, employment_quarterly[:, funds_idx], funds_quarterly):
            w.writerow([q, f"{qcew_v:.12g}", f"{ces_v:.12g}", f"{ces_v / qcew_v:.12g}"])

    with (RESULTS_DIR / "target_employment_corrections.csv").open("w", newline="", encoding="utf-8-sig") as f:
        fields = ["node_key", "default_source", "replacement_source", "reason", "formula_changed"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows([
            {
                "node_key": "rail_transport",
                "default_source": "BLS QCEW NAICS 482",
                "replacement_source": "BLS CES CEU4348200001, Rail Transportation all employees, NSA",
                "reason": "QCEW structurally excludes railroad workers covered by the railroad unemployment insurance system",
                "formula_changed": "False",
            },
            {
                "node_key": "funds_trusts",
                "default_source": "BLS QCEW NAICS 525",
                "replacement_source": "BLS CES CEU5552300001, Securities/Investments + Funds/Trusts aggregate all employees, NSA",
                "reason": "BLS 2013 reclassification breaks standalone NAICS 525 employment; use the continuous 523+525 CES aggregate and match the productivity numerator to the same BEA aggregate",
                "formula_changed": "False",
            },
        ])

    qcew_cache = RAW_BLS_DIR / f"qcew_monthly_employment_{HISTORY_START_YEAR}_{latest_gos_year}.json"
    manifest.append({
        "source": "BLS",
        "dataset": "QCEW monthly employment",
        "url": BLS_API_URL,
        "path": str(qcew_cache.relative_to(ROOT)),
        "bytes": qcew_cache.stat().st_size,
        "sha256": sha256_file(qcew_cache),
    })
    manifest.append({
        "source": "BLS",
        "dataset": "CES Rail Transportation all employees, NSA (CEU4348200001)",
        "url": BLS_CES_FLAT_URL,
        "path": str(rail_cache.relative_to(ROOT)),
        "bytes": rail_cache.stat().st_size,
        "sha256": sha256_file(rail_cache),
    })
    manifest.append({
        "source": "BLS",
        "dataset": "CES Securities/Investments + Funds/Trusts aggregate, all employees, NSA (CEU5552300001)",
        "url": BLS_CES_FLAT_URL,
        "path": str(funds_cache.relative_to(ROOT)),
        "bytes": funds_cache.stat().st_size,
        "sha256": sha256_file(funds_cache),
    })
    ces_vintage_cache = RAW_BLS_DIR / "cesvinall.zip"
    if ces_vintage_cache.exists():
        manifest.append({
            "source": "BLS",
            "dataset": "CES all-employees publication-vintage archive (cesvinall.zip)",
            "url": "https://www.bls.gov/web/empsit/cesvinall.zip",
            "path": str(ces_vintage_cache.relative_to(ROOT)),
            "bytes": ces_vintage_cache.stat().st_size,
            "sha256": sha256_file(ces_vintage_cache),
        })
    ces_industry_cache = RAW_BLS_DIR / "ce.industry"
    if ces_industry_cache.exists():
        manifest.append({
            "source": "BLS",
            "dataset": "CES industry metadata used for exact node mapping",
            "url": "https://download.bls.gov/pub/time.series/ce/ce.industry",
            "path": str(ces_industry_cache.relative_to(ROOT)),
            "bytes": ces_industry_cache.stat().st_size,
            "sha256": sha256_file(ces_industry_cache),
        })

    with (RESULTS_DIR / "source_manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["source", "dataset", "url", "path", "bytes", "sha256"])
        w.writeheader()
        w.writerows(manifest)

    summary = {
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "node_count": EXPECTED_NODE_COUNT,
        "bea_quarterly_available": [q_cva[0], q_cva[-1]],
        "bea_annual_gos_available": [gos_years_all[0], gos_years_all[-1]],
        "target_collection_period": [f"{TARGET_START_YEAR}Q1", target_quarter_end],
        "target_quarter_count": len(target_quarters),
        "qcew_monthly_period": [months[0], months[-1]],
        "qcew_month_count": len(months),
        "bea_direct_nodes": sum(r["mapping"] == "DIRECT" for r in mapping_cva),
        "bea_composite_nodes": sum(r["mapping"] == "COMPOSITE_SUM" for r in mapping_cva),
        "qcew_complete_nodes": EXPECTED_NODE_COUNT,
        "history_annual_period": [history_years[0], history_years[-1]],
        "target_release_metadata": str(RELEASE_METADATA_CSV.relative_to(ROOT)),
        "point_in_time_history_raw": str(PIT_HISTORY_RAW_CSV.relative_to(ROOT)),
        "point_in_time_history_audit": str(PIT_HISTORY_AUDIT_CSV.relative_to(ROOT)),
        "point_in_time_history": pit_history_summary,
        "employment_target_rule": {
            "exact_ces_nodes": int(len(ces_exact_final_monthly)),
            "exact_ces": "BLS CES current-final NSA employment where an exact publication-vintage industry mapping exists; PIT uses the corresponding historical publication vintage",
            "qcew_fallback_nodes": int(EXPECTED_NODE_COUNT - len(ces_exact_final_monthly) - 1),
            "qcew_fallback": "BLS QCEW quarterly-average employment for nodes without an exact CES publication-vintage mapping",
            "rail_transport": "Exact BLS CES Rail Transportation employment; avoids QCEW railroad exclusion",
            "funds_trusts": "BLS CES CEU5552300001 quarterly-average 523+525 employment proxy; productivity numerator uses matching BEA securities+funds real VA aggregate",
        },
        "required_inputs": {
            "profitability": [
                "annual_gos",
                "current_gross_output",
                "current_value_added_for_temporal_disaggregation",
                "4Q/year-over-year change",
            ],
            "productivity": [
                "within-vintage real_value_added per worker 4Q symmetric percentage change",
                "employment 4Q/year-over-year ratio",
            ],
            "growth": ["real_gross_output 4Q/year-over-year growth"],
        },
        "status": "PASS",
    }
    (RESULTS_DIR / "collection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("[5/5] Validation PASS")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


# ---- Point-in-time vintage/history implementation (Stage 01) ----

BEA_ARCHIVE_API = "https://apps.bea.gov/histdata/core/data"
BEA_ARCHIVE_FILES = "https://apps.bea.gov/HistData/Files"
PARSED_SNAPSHOT_CACHE_VERSION = 2
CES_VINTAGE_URL = "https://www.bls.gov/web/empsit/cesvinall.zip"
CES_INDUSTRY_URL = "https://download.bls.gov/pub/time.series/ce/ce.industry"
FUNDS_VINTAGE_FILE = "tri_555230_NSA.csv"  # continuous CES 523+525 aggregate proxy

CES_NAME_FALLBACK = {
    "retail": "Retail trade",
    "wholesale": "Wholesale trade",
    "information": "Information",
    "other_services": "Other services",
}


MONTH_ABBR = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}


PARENT_GOS = {
    "oil_gas_extraction": "Mining",
    "mining_except_oil_gas": "Mining",
    "support_mining": "Mining",
    "wood_products": "Manufacturing",
    "nonmetallic_mineral": "Manufacturing",
    "primary_metals": "Manufacturing",
    "fabricated_metal": "Manufacturing",
    "machinery": "Manufacturing",
    "computer_electronic": "Manufacturing",
    "electrical_equipment": "Manufacturing",
    "motor_vehicles": "Manufacturing",
    "other_transport_equipment": "Manufacturing",
    "furniture": "Manufacturing",
    "misc_manufacturing": "Manufacturing",
    "food_beverage_tobacco": "Manufacturing",
    "textiles": "Manufacturing",
    "apparel_leather": "Manufacturing",
    "paper": "Manufacturing",
    "printing": "Manufacturing",
    "petroleum_coal": "Manufacturing",
    "chemicals": "Manufacturing",
    "plastics_rubber": "Manufacturing",
    "air_transport": "Transportation and warehousing",
    "rail_transport": "Transportation and warehousing",
    "water_transport": "Transportation and warehousing",
    "truck_transport": "Transportation and warehousing",
    "transit_ground": "Transportation and warehousing",
    "pipeline": "Transportation and warehousing",
    "other_transport_support": "Transportation and warehousing",
    "warehousing": "Transportation and warehousing",
    "credit_intermediation": "Finance and insurance",
    "securities": "Finance and insurance",
    "insurance": "Finance and insurance",
    "funds_trusts": "Finance and insurance",
    "real_estate": "Real estate and rental and leasing",
    "rental_leasing": "Real estate and rental and leasing",
    "legal": "Professional, scientific, and technical services",
    "computer_systems": "Professional, scientific, and technical services",
    "misc_professional": "Professional, scientific, and technical services",
    "admin_support": "Administrative and waste management services",
    "waste_remediation": "Administrative and waste management services",
    "ambulatory_health": "Health care and social assistance",
    "hospitals_nursing": "Health care and social assistance",
    "social_assistance": "Health care and social assistance",
    "performing_arts_museums": "Arts, entertainment, and recreation",
    "amusement_recreation": "Arts, entertainment, and recreation",
    "accommodation": "Accommodation and food services",
    "food_services": "Accommodation and food services",
}


def _clean_name(value: object) -> str:
    s = str(value).replace("…", " ").replace("\xa0", " ").strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*/\d+/\s*$", "", s)
    return s


def _key(value: object) -> str:
    return _clean_name(value).casefold()


def _parse_number(value: object) -> float | None:
    s = str(value).replace(",", "").strip()
    if not s or s in {"...", "—", "--", "nan", "None"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _xls_matrix(data: bytes, sheet_name: str) -> list[list[object]]:
    wb = xlrd.open_workbook(file_contents=data)
    if sheet_name not in wb.sheet_names():
        raise KeyError(sheet_name)
    sh = wb.sheet_by_name(sheet_name)
    return [[sh.cell_value(r, c) for c in range(sh.ncols)] for r in range(sh.nrows)]


def _xlsx_matrix(data: bytes, sheet_name: str) -> list[list[object]]:
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        wb = ET.fromstring(z.read("xl/workbook.xml"))
        rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        ridmap = {x.attrib["Id"]: x.attrib["Target"] for x in rels}
        shared: list[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            ss = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in ss.findall("m:si", ns):
                shared.append("".join(t.text or "" for t in si.iterfind(".//m:t", ns)))

        target = None
        for sheet in wb.findall("m:sheets/m:sheet", ns):
            if sheet.attrib["name"] == sheet_name:
                rid = sheet.attrib[
                    "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
                ]
                target = ridmap[rid]
                break
        if target is None:
            raise KeyError(sheet_name)
        sheet_path = target.lstrip("/") if target.startswith("/") else str(PurePosixPath("xl") / target)
        root = ET.fromstring(z.read(sheet_path))

        def col_num(ref: str) -> int:
            m = re.match(r"([A-Z]+)", ref)
            if not m:
                raise ValueError(ref)
            n = 0
            for ch in m.group(1):
                n = n * 26 + ord(ch) - 64
            return n - 1

        sparse: list[dict[int, object]] = []
        max_col = 0
        for row in root.findall(".//m:sheetData/m:row", ns):
            out: dict[int, object] = {}
            for cell in row.findall("m:c", ns):
                idx = col_num(cell.attrib.get("r", "A1"))
                max_col = max(max_col, idx)
                typ = cell.attrib.get("t")
                v = cell.find("m:v", ns)
                inline = cell.find("m:is", ns)
                if typ == "s" and v is not None:
                    value: object = shared[int(v.text)]
                elif typ == "inlineStr" and inline is not None:
                    value = "".join(t.text or "" for t in inline.iterfind(".//m:t", ns))
                elif v is not None:
                    value = v.text or ""
                else:
                    value = ""
                out[idx] = value
            sparse.append(out)
        return [[row.get(c, "") for c in range(max_col + 1)] for row in sparse]


def _matrix(data: bytes, filename: str, sheet_name: str) -> list[list[object]]:
    if filename.lower().endswith(".xlsx"):
        return _xlsx_matrix(data, sheet_name)
    return _xls_matrix(data, sheet_name)


def _pick_xls_sheet(data: bytes, *candidates: str) -> str:
    wb = xlrd.open_workbook(file_contents=data, on_demand=True)
    names = wb.sheet_names()
    for candidate in candidates:
        if candidate in names:
            return candidate
    raise KeyError(f"None of {candidates} found; available={names}")


def _annual_table(matrix: list[list[object]]) -> dict[str, dict[int, float]]:
    header_idx = None
    year_cols: list[tuple[int, int]] = []
    for i, row in enumerate(matrix):
        cols = []
        for c, value in enumerate(row):
            s = str(value).strip()
            if re.fullmatch(r"(?:19|20)\d{2}(?:\.0)?", s):
                cols.append((c, int(float(s))))
        if len(cols) >= 2:
            header_idx = i
            year_cols = cols
            break
    if header_idx is None:
        raise RuntimeError("Could not locate annual header")
    out: dict[str, dict[int, float]] = {}
    for row in matrix[header_idx + 1 :]:
        if len(row) < 2:
            continue
        name = _clean_name(row[1])
        if not name:
            continue
        vals: dict[int, float] = {}
        for c, year in year_cols:
            if c >= len(row):
                continue
            num = _parse_number(row[c])
            if num is not None:
                vals[year] = num
        if vals:
            out[name] = vals
    return out


def _value_unit_multiplier(matrix: list[list[object]]) -> float:
    """Convert BEA dollar levels to the current pipeline's millions-of-dollars unit."""
    head = " ".join(
        str(cell) for row in matrix[:6] for cell in row[:6]
    ).casefold()
    if "billions of" in head:
        return 1000.0
    return 1.0


def _scale_table(table: dict[str, dict], factor: float) -> dict[str, dict]:
    if factor == 1.0:
        return table
    return {
        name: {period: float(value) * factor for period, value in values.items()}
        for name, values in table.items()
    }


def _quarterly_table(matrix: list[list[object]]) -> dict[str, dict[str, float]]:
    # Current BEA GDP-by-Industry workbooks use a single header row with direct
    # YYYYQn labels (for example 2005Q1, 2005Q2, ...). Older archive workbooks
    # used a year row followed by I/II/III/IV. Support both layouts.
    direct_idx = None
    direct_cols: list[tuple[int, str]] = []
    for i, row in enumerate(matrix):
        cols = []
        for c, value in enumerate(row):
            token = str(value).strip().upper()
            if re.fullmatch(r"(?:19|20)\d{2}Q[1-4]", token):
                cols.append((c, token))
        if len(cols) >= 4:
            direct_idx = i
            direct_cols = cols
            break
    if direct_idx is not None:
        out: dict[str, dict[str, float]] = {}
        for row in matrix[direct_idx + 1 :]:
            if len(row) < 2:
                continue
            name = _clean_name(row[1])
            if not name:
                continue
            vals: dict[str, float] = {}
            for c, period in direct_cols:
                if c >= len(row):
                    continue
                num = _parse_number(row[c])
                if num is not None:
                    vals[period] = num
            if vals:
                out[name] = vals
        if out:
            return out

    qrow_idx = None
    for i, row in enumerate(matrix):
        tokens = [str(x).strip().upper() for x in row]
        if sum(x in {"I", "II", "III", "IV"} for x in tokens) >= 4:
            qrow_idx = i
            break
    if qrow_idx is None:
        raise RuntimeError("Could not locate quarterly header")

    year_row = matrix[qrow_idx - 1]
    years: list[int | None] = []
    current: int | None = None
    for value in year_row:
        s = str(value).strip()
        if re.fullmatch(r"(?:19|20)\d{2}(?:\.0)?", s):
            current = int(float(s))
        years.append(current)

    roman = {"I": 1, "II": 2, "III": 3, "IV": 4}
    period_cols: list[tuple[int, str]] = []
    for c, value in enumerate(matrix[qrow_idx]):
        token = str(value).strip().upper()
        if token in roman and c < len(years) and years[c] is not None:
            period_cols.append((c, f"{years[c]}Q{roman[token]}"))
    if not period_cols:
        raise RuntimeError("No quarterly periods parsed")

    out: dict[str, dict[str, float]] = {}
    for row in matrix[qrow_idx + 1 :]:
        if len(row) < 2:
            continue
        name = _clean_name(row[1])
        if not name:
            continue
        vals: dict[str, float] = {}
        for c, period in period_cols:
            if c >= len(row):
                continue
            num = _parse_number(row[c])
            if num is not None:
                vals[period] = num
        if vals:
            out[name] = vals
    return out


def _gos_table(matrix: list[list[object]]) -> dict[str, dict[int, float]]:
    base = _annual_table(matrix)
    # Annual parser alone cannot associate repeated component rows with their
    # parent industry. Re-read the header and walk the rows explicitly.
    header_idx = None
    year_cols: list[tuple[int, int]] = []
    for i, row in enumerate(matrix):
        cols = []
        for c, value in enumerate(row):
            s = str(value).strip()
            if re.fullmatch(r"(?:19|20)\d{2}(?:\.0)?", s):
                cols.append((c, int(float(s))))
        if len(cols) >= 2:
            header_idx, year_cols = i, cols
            break
    if header_idx is None:
        return {}
    components = {
        "compensation of employees",
        "taxes on production and imports less subsidies",
        "gross operating surplus",
    }
    current_industry: str | None = None
    out: dict[str, dict[int, float]] = {}
    for row in matrix[header_idx + 1 :]:
        if len(row) < 2:
            continue
        name = _clean_name(row[1])
        if not name:
            continue
        lower = name.casefold()
        if lower not in components:
            current_industry = name
            continue
        if lower != "gross operating surplus" or current_industry is None:
            continue
        vals: dict[int, float] = {}
        for c, year in year_cols:
            if c >= len(row):
                continue
            num = _parse_number(row[c])
            if num is not None:
                vals[year] = num
        if vals:
            out[current_industry] = vals
    return out


def _lookup(table: dict[str, dict], name: str) -> dict | None:
    wanted = _key(name)
    for actual, values in table.items():
        if _key(actual) == wanted:
            return values
    return None


def _node_name(node: dict[str, str], alias: dict[str, str]) -> str:
    return alias.get(node["key"], node["name"])


def _node_values(table: dict[str, dict], node: dict[str, str], alias: dict[str, str]) -> dict | None:
    key = node["key"]
    if key == "hospitals_nursing":
        combined = _lookup(table, "Hospitals and nursing and residential care facilities")
        if combined is not None:
            return combined
        h = _lookup(table, "Hospitals")
        n = _lookup(table, "Nursing and residential care facilities")
        if h is not None and n is not None:
            common = set(h) & set(n)
            return {p: float(h[p]) + float(n[p]) for p in common}
        return None
    return _lookup(table, _node_name(node, alias))


def _bea_release_date_from_path(path: str) -> pd.Timestamp | None:
    last = path.replace("/", "\\").split("\\")[-1]
    m = re.search(
        r"(January|February|March|April|May|June|July|August|September|October|November|December)-"
        r"(\d{1,2})-(\d{4})$",
        last,
        re.I,
    )
    if not m:
        return None
    return pd.Timestamp(f"{m.group(1)} {m.group(2)}, {m.group(3)}").normalize()


def _archive_leaf_paths(session: requests.Session) -> dict[pd.Timestamp, list[str]]:
    obj = session.get(
        f"{BEA_ARCHIVE_API}/Fea_DisplayChildrenC/",
        params={"HistMainId": 8, "getFiles": "false", "getDirs": "true"},
        timeout=60,
    ).json()
    grouped: dict[pd.Timestamp, list[str]] = {}
    for path in obj.get("FileArray", []):
        if "GDP_by_Industry" not in path or path.endswith("_notes"):
            continue
        d = _bea_release_date_from_path(path)
        if d is None:
            continue
        grouped.setdefault(d, []).append(path)
    return grouped


def _list_archive_files(session: requests.Session, path: str) -> list[str]:
    ids = session.get(
        f"{BEA_ARCHIVE_API}/UrlPath_getID/", params={"UrlPath": path}, timeout=60
    ).json()
    if not ids:
        return []
    the_id = ids[0].get("Theid")
    if not the_id:
        return []
    resolved = session.get(f"{BEA_ARCHIVE_API}/getPath/{the_id}", timeout=60).json()
    if not resolved:
        return []
    resolved_path = resolved[0].get("Thepath")
    if not resolved_path:
        return []
    obj = session.get(
        f"{BEA_ARCHIVE_API}/Fea_DisplayChildrenC/",
        params={"HistMainId": 8, "thePath": resolved_path, "getFiles": "true", "getDirs": "false"},
        timeout=60,
    ).json()
    return obj.get("Filearray3", []) or []


def _public_archive_url(file_path: str) -> str:
    p = file_path.replace("\\", "/")
    prefix = "/Inetpub/wwwroot/website/website/HistData/Files/"
    if p.startswith(prefix):
        p = p[len(prefix) :]
    return f"{BEA_ARCHIVE_FILES}/{p}"


def _download_bytes(session: requests.Session, url: str, cache: Path, refresh: bool) -> bytes:
    if refresh or not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        r = session.get(url, timeout=120)
        r.raise_for_status()
        cache.write_bytes(r.content)
    return cache.read_bytes()


def _snapshot_from_release(
    session: requests.Session,
    release_date: pd.Timestamp,
    candidate_paths: list[str],
    cache_dir: Path,
    refresh: bool,
) -> dict | None:
    # Prefer the deepest archive path; older annual releases often expose a
    # duplicate parent and an /Annual/ child, with files only under the child.
    for path in sorted(candidate_paths, key=lambda x: (-x.count("\\"), x)):
        files = _list_archive_files(session, path)
        names = {x.replace("\\", "/").split("/")[-1]: x for x in files}
        if not names:
            continue

        annual_zip_name = "AllTables.zip" if "AllTables.zip" in names else None
        qtr_zip_name = "AllTablesQTR.zip" if "AllTablesQTR.zip" in names else None
        legacy_zip_name = next((n for n in names if re.fullmatch(r"GDPbyInd\d{4}\.zip", n, re.I)), None)
        modern_zip_name = next((n for n in names if re.fullmatch(r"GdpByInd\.zip", n, re.I)), None)
        if annual_zip_name is None and legacy_zip_name is None and modern_zip_name is None:
            continue

        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{release_date.date()}_{path.split('\\')[-1]}")
        parsed_cache = (
            cache_dir / safe / f"parsed_snapshot_v{PARSED_SNAPSHOT_CACHE_VERSION}.pkl"
        )
        if parsed_cache.exists() and not refresh:
            try:
                with parsed_cache.open("rb") as f:
                    cached = pickle.load(f)
                if isinstance(cached, dict) and cached.get("release_date") == release_date:
                    return cached
            except Exception:
                parsed_cache.unlink(missing_ok=True)
        out: dict = {
            "release_date": release_date,
            "archive_path": path,
            "annual": {},
            "quarterly": {},
        }

        if modern_zip_name is not None:
            raw = _download_bytes(
                session,
                _public_archive_url(names[modern_zip_name]),
                cache_dir / safe / modern_zip_name,
                refresh,
            )
            z = zipfile.ZipFile(io.BytesIO(raw))
            va_name = next(
                n for n in z.namelist()
                if re.search(r"(?:^|/)valueadded\.xlsx$", n, re.I)
            )
            go_name = next(
                n for n in z.namelist()
                if re.search(r"(?:^|/)grossoutput\.xlsx$", n, re.I)
            )
            va = z.read(va_name)
            go = z.read(go_name)

            annual_va_matrix = _matrix(va, va_name, "TVA105-A")
            annual_real_va_matrix = _matrix(va, va_name, "TVA106-A")
            gos_matrix = _matrix(va, va_name, "TVA113-A")
            annual_go_matrix = _matrix(go, go_name, "TGO105-A")
            annual_real_go_matrix = _matrix(go, go_name, "TGO106-A")
            out["annual"]["current_va"] = _annual_table(annual_va_matrix)
            out["annual"]["real_va"] = _scale_table(
                _annual_table(annual_real_va_matrix),
                _value_unit_multiplier(annual_real_va_matrix),
            )
            out["annual"]["gos"] = _gos_table(gos_matrix)
            out["annual"]["current_go"] = _annual_table(annual_go_matrix)
            out["annual"]["real_go"] = _annual_table(annual_real_go_matrix)

            q_va_matrix = _matrix(va, va_name, "TVA105-Q")
            q_real_va_matrix = _matrix(va, va_name, "TVA106-Q")
            q_go_matrix = _matrix(go, go_name, "TGO105-Q")
            q_real_go_matrix = _matrix(go, go_name, "TGO106-Q")
            out["quarterly"]["current_va"] = _quarterly_table(q_va_matrix)
            out["quarterly"]["real_va"] = _scale_table(
                _quarterly_table(q_real_va_matrix),
                _value_unit_multiplier(q_real_va_matrix),
            )
            out["quarterly"]["current_go"] = _quarterly_table(q_go_matrix)
            out["quarterly"]["real_go"] = _quarterly_table(q_real_go_matrix)
        elif annual_zip_name is not None:
            raw = _download_bytes(
                session,
                _public_archive_url(names[annual_zip_name]),
                cache_dir / safe / annual_zip_name,
                refresh,
            )
            z = zipfile.ZipFile(io.BytesIO(raw))
            va_name = next(
                n for n in z.namelist()
                if re.search(r"(?:^|/)valueadded\.xls[x]?$", n, re.I)
            )
            go_name = next(
                n for n in z.namelist()
                if re.search(r"(?:^|/)grossoutput\.xls[x]?$", n, re.I)
            )
            va = z.read(va_name)
            go = z.read(go_name)
            if va_name.lower().endswith(".xls"):
                va_sheet = _pick_xls_sheet(va, "VA", "Cu$")
                components_sheet = _pick_xls_sheet(va, "Components", "ComponentsCu$")
                go_sheet = _pick_xls_sheet(go, "GO", "CurrentDollars")
            else:
                va_sheet = "VA"
                components_sheet = "Components"
                go_sheet = "GO"
            annual_va_matrix = _matrix(va, va_name, va_sheet)
            annual_real_va_matrix = _matrix(va, va_name, "RealVA")
            components_matrix = _matrix(va, va_name, components_sheet)
            out["annual"]["current_va"] = _annual_table(annual_va_matrix)
            out["annual"]["real_va"] = _scale_table(
                _annual_table(annual_real_va_matrix),
                _value_unit_multiplier(annual_real_va_matrix),
            )
            out["annual"]["gos"] = _gos_table(components_matrix)
            out["annual"]["current_go"] = _annual_table(_matrix(go, go_name, go_sheet))
            try:
                out["annual"]["real_go"] = _annual_table(_matrix(go, go_name, "Real GO"))
            except KeyError:
                out["annual"]["go_qty"] = _annual_table(_matrix(go, go_name, "ChainQtyIndexes"))
        else:
            assert legacy_zip_name is not None
            raw = _download_bytes(
                session,
                _public_archive_url(names[legacy_zip_name]),
                cache_dir / safe / legacy_zip_name,
                refresh,
            )
            z = zipfile.ZipFile(io.BytesIO(raw))
            va_name = next(n for n in z.namelist() if n.lower() == "valueadded.xls")
            go_name = next(n for n in z.namelist() if n.lower() == "grossoutput.xls")
            va = z.read(va_name)
            go = z.read(go_name)
            va_sheet = _pick_xls_sheet(va, "VA", "Cu$")
            components_sheet = _pick_xls_sheet(va, "Components", "ComponentsCu$")
            go_sheet = _pick_xls_sheet(go, "GO", "CurrentDollars")
            annual_va_matrix = _matrix(va, va_name, va_sheet)
            annual_real_va_matrix = _matrix(va, va_name, "RealVA")
            components_matrix = _matrix(va, va_name, components_sheet)
            out["annual"]["current_va"] = _annual_table(annual_va_matrix)
            out["annual"]["real_va"] = _scale_table(
                _annual_table(annual_real_va_matrix),
                _value_unit_multiplier(annual_real_va_matrix),
            )
            out["annual"]["gos"] = _gos_table(components_matrix)
            out["annual"]["current_go"] = _annual_table(_matrix(go, go_name, go_sheet))
            out["annual"]["go_qty"] = _annual_table(_matrix(go, go_name, "ChainQtyIndexes"))

        if qtr_zip_name is not None:
            raw = _download_bytes(
                session,
                _public_archive_url(names[qtr_zip_name]),
                cache_dir / safe / qtr_zip_name,
                refresh,
            )
            z = zipfile.ZipFile(io.BytesIO(raw))
            va_name = next(n for n in z.namelist() if "valueaddedqtr" in n.casefold())
            go_name = next(n for n in z.namelist() if "grossoutputqtr" in n.casefold())
            va = z.read(va_name)
            go = z.read(go_name)
            q_va_matrix = _matrix(va, va_name, "VA")
            q_real_va_matrix = _matrix(va, va_name, "RealVA")
            out["quarterly"]["current_va"] = _quarterly_table(q_va_matrix)
            out["quarterly"]["real_va"] = _scale_table(
                _quarterly_table(q_real_va_matrix),
                _value_unit_multiplier(q_real_va_matrix),
            )
            out["quarterly"]["current_go"] = _quarterly_table(_matrix(go, go_name, "GO"))
            try:
                out["quarterly"]["real_go"] = _quarterly_table(_matrix(go, go_name, "Real GO"))
            except KeyError:
                out["quarterly"]["go_qty"] = _quarterly_table(_matrix(go, go_name, "ChainQtyIndexes"))
        parsed_cache.parent.mkdir(parents=True, exist_ok=True)
        with parsed_cache.open("wb") as f:
            pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
        return out
    return None


def _latest_snapshot(
    cutoff: pd.Timestamp,
    releases: dict[pd.Timestamp, list[str]],
    cache: dict[pd.Timestamp, dict | None],
    session: requests.Session,
    cache_dir: Path,
    refresh: bool,
) -> dict:
    for date in sorted((d for d in releases if d <= cutoff), reverse=True):
        if date not in cache:
            cache[date] = _snapshot_from_release(session, date, releases[date], cache_dir, refresh)
        if cache[date] is not None:
            snapshot = cache[date]
            # Origins are processed chronologically. Keeping every parsed BEA
            # vintage retains millions of Python objects and eventually exhausts
            # memory, while only the current/latest successful snapshot can be
            # reused by the next origin. Preserve failed sentinels but release
            # all older successful snapshots immediately.
            stale = [k for k, v in cache.items() if k != date and v is not None]
            for k in stale:
                del cache[k]
            return snapshot  # type: ignore[return-value]
    raise RuntimeError(f"No usable BEA vintage available by {cutoff.date()}")


def _series_value(table: dict[str, dict], node: dict[str, str], alias: dict[str, str], period) -> float | None:
    series = _node_values(table, node, alias)
    if series is None or period not in series:
        return None
    return float(series[period])


def _growth_value(
    snapshot: dict,
    resolution: str,
    node: dict[str, str],
    alias: dict[str, str],
    period,
    previous,
) -> float | None:
    block = snapshot[resolution]
    table = block.get("real_go") or block.get("go_qty")
    if table is None:
        return None
    cur = _series_value(table, node, alias, period)
    prev = _series_value(table, node, alias, previous)
    if cur is None or prev is None or abs(prev) <= 1e-15:
        return None
    return cur / prev - 1.0


def _profitability(
    snapshot: dict,
    resolution: str,
    node: dict[str, str],
    alias: dict[str, str],
    period,
) -> tuple[float | None, str]:
    annual = snapshot["annual"]
    if resolution == "annual":
        va_period = int(period)
        cva = _series_value(annual["current_va"], node, alias, va_period)
        cgo = _series_value(annual["current_go"], node, alias, va_period)
    else:
        cva = _series_value(snapshot["quarterly"]["current_va"], node, alias, period)
        cgo = _series_value(snapshot["quarterly"]["current_go"], node, alias, period)
    if cva is None or cgo is None or abs(cgo) <= 1e-15:
        return None, "MISSING"

    gos_table = annual.get("gos", {})
    exact = _node_values(gos_table, node, alias)
    gos_years = set(exact) if exact else set()
    if resolution == "annual" and int(period) in gos_years:
        return float(exact[int(period)]) / cgo, "EXACT"

    # For quarterly state, or older vintages without detailed GOS components,
    # use the latest GOS/VA share actually published in that vintage.
    source_name = _node_name(node, alias)
    gos_series = exact
    va_series = _node_values(annual["current_va"], node, alias)
    mode = "EXACT_SHARE"
    if gos_series is None or va_series is None or not (set(gos_series) & set(va_series)):
        parent = PARENT_GOS.get(node["key"], source_name)
        gos_series = _lookup(gos_table, parent)
        va_series = _lookup(annual["current_va"], parent)
        mode = f"PARENT_SHARE:{parent}"
    if gos_series is None or va_series is None:
        return None, "MISSING"
    common = sorted(set(gos_series) & set(va_series))
    if not common:
        return None, "MISSING"
    if resolution == "annual":
        eligible = [y for y in common if y <= int(period)]
    else:
        eligible = [y for y in common if y <= pd.Period(period, freq="Q").year]
    if not eligible:
        return None, "MISSING"
    year = max(eligible)
    if abs(float(va_series[year])) <= 1e-15:
        return None, "MISSING"
    share = float(gos_series[year]) / float(va_series[year])
    return share * cva / cgo, mode


def _load_ces_vintage_file(
    session: requests.Session,
    cache_path: Path,
    refresh: bool,
    filename: str,
) -> pd.DataFrame:
    if refresh or not cache_path.exists() or cache_path.stat().st_size < 100_000:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        r = session.get(CES_VINTAGE_URL, headers={"User-Agent": "Mozilla/5.0 PRISM academic research"}, timeout=120)
        r.raise_for_status()
        cache_path.write_bytes(r.content)
    with zipfile.ZipFile(cache_path) as z:
        return pd.read_csv(io.BytesIO(z.read(filename)))


def _expand_compact_naics(spec: str) -> set[str]:
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


def _node_naics_set(node: dict[str, str]) -> set[str]:
    return {x.strip() for x in node.get("naics_codes", "").split(",") if x.strip()}


def build_ces_exact_vintage_mapping(
    session: requests.Session,
    nodes: list[dict[str, str]],
    raw_bls_dir: Path,
    refresh: bool,
) -> pd.DataFrame:
    """Deterministically map PRISM nodes to exact CES publication-vintage files."""
    industry_cache = raw_bls_dir / "ce.industry"
    if refresh or not industry_cache.exists():
        sibling_cache = (
            raw_bls_dir.parents[2]
            / "02_initial_feature_pool"
            / "raw"
            / "BLS"
            / "ce.industry"
        )
        industry_cache.parent.mkdir(parents=True, exist_ok=True)
        if sibling_cache.exists() and not refresh:
            industry_cache.write_bytes(sibling_cache.read_bytes())
        else:
            r = session.get(CES_INDUSTRY_URL, timeout=120)
            r.raise_for_status()
            industry_cache.write_text(r.text, encoding="utf-8")
    industry = pd.read_csv(industry_cache, sep="\t", dtype=str).fillna("")
    industry.columns = [c.strip() for c in industry.columns]
    for c in industry.columns:
        industry[c] = industry[c].astype(str).str.strip()

    ces_cache = raw_bls_dir / "cesvinall.zip"
    if refresh or not ces_cache.exists() or ces_cache.stat().st_size < 100_000:
        r = session.get(
            CES_VINTAGE_URL,
            headers={"User-Agent": "Mozilla/5.0 PRISM academic research"},
            timeout=120,
        )
        r.raise_for_status()
        ces_cache.parent.mkdir(parents=True, exist_ok=True)
        ces_cache.write_bytes(r.content)
    with zipfile.ZipFile(ces_cache) as z:
        vintage_files = set(z.namelist())

    industry = industry.copy()
    industry["naics_set"] = industry["naics_code"].map(_expand_compact_naics)
    industry["publication_code"] = industry["industry_code"].str[:6]
    industry["vintage_filename"] = "tri_" + industry["publication_code"] + "_NSA.csv"
    industry["has_vintage"] = industry["vintage_filename"].isin(vintage_files)

    rows = []
    for node in nodes:
        target = _node_naics_set(node)
        hits = industry[
            industry["naics_code"].map(lambda x: _expand_compact_naics(x) == target)
        ]
        mapping_mode = "NAICS_EXACT"
        if hits.empty:
            wanted = CES_NAME_FALLBACK.get(node["key"])
            if wanted:
                hits = industry[
                    industry["industry_name"].str.casefold() == wanted.casefold()
                ]
                mapping_mode = "NAME_FALLBACK"
        if hits.empty:
            code = ""
            name = ""
        else:
            h = hits.copy()
            h["display_num"] = pd.to_numeric(
                h["display_level"], errors="coerce"
            ).fillna(99)
            best = h.sort_values("display_num").iloc[0]
            code = str(best["industry_code"])
            name = str(best["industry_name"])
        publication_code = code[:6] if code else ""
        filename = f"tri_{publication_code}_NSA.csv" if publication_code else ""

        direct_ok = bool(filename and filename in vintage_files)
        codes: list[str] = [code] if direct_ok else []
        pub_codes: list[str] = [publication_code] if direct_ok else []
        names_used: list[str] = [name] if direct_ok else []
        files_used: list[str] = [filename] if direct_ok else []
        component_count = 1 if direct_ok else 0

        if not direct_ok and target:
            candidates = industry[
                industry["has_vintage"]
                & industry["naics_set"].map(
                    lambda s: bool(s) and s.issubset(target)
                )
            ].copy()
            candidates["set_key"] = candidates["naics_set"].map(
                lambda s: "|".join(sorted(s))
            )
            candidates["display_num"] = pd.to_numeric(
                candidates["display_level"], errors="coerce"
            ).fillna(99)
            candidates = (
                candidates.sort_values(["display_num", "industry_code"])
                .drop_duplicates("set_key")
            )
            cand = list(candidates.itertuples(index=False))
            solution = None
            for size in range(1, min(6, len(cand)) + 1):
                for combo in itertools.combinations(cand, size):
                    sets = [x.naics_set for x in combo]
                    union = set().union(*sets)
                    if union != target:
                        continue
                    if sum(len(s) for s in sets) != len(union):
                        continue
                    solution = combo
                    break
                if solution is not None:
                    break
            if solution is not None:
                mapping_mode = "NAICS_EXACT_COMPOSITE"
                codes = [str(x.industry_code) for x in solution]
                pub_codes = [str(x.publication_code) for x in solution]
                names_used = [str(x.industry_name) for x in solution]
                files_used = [str(x.vintage_filename) for x in solution]
                component_count = len(solution)

        rows.append({
            "node_key": node["key"],
            "node_name": node["name"],
            "mapping_mode": mapping_mode if files_used else "NONE",
            "component_count": component_count,
            "ces_industry_code": ";".join(codes),
            "ces_publication_code": ";".join(pub_codes),
            "ces_industry_name": ";".join(names_used),
            "vintage_filename": ";".join(files_used),
            "vintage_file_exists": bool(files_used),
        })
    return pd.DataFrame(rows)


def _compact_ces_vintage(df: pd.DataFrame, min_year: int, max_year: int) -> pd.DataFrame:
    wanted = ["year", "month"]
    for year in range(min_year, max_year + 1):
        for month, abbr in MONTH_ABBR.items():
            col = f"{abbr}_{year % 100:02d}"
            if col in df.columns:
                wanted.append(col)
    out = df[wanted].copy()
    out = out[(out["year"] >= min_year) & (out["year"] <= max_year + 1)].reset_index(drop=True)
    return out


def load_exact_ces_vintages(
    session: requests.Session,
    mapping: pd.DataFrame,
    raw_bls_dir: Path,
    refresh: bool,
    min_year: int,
    max_year: int,
) -> dict[str, pd.DataFrame]:
    cache = raw_bls_dir / "cesvinall.zip"
    out: dict[str, pd.DataFrame] = {}
    first = True
    for r in mapping.itertuples(index=False):
        if not bool(r.vintage_file_exists):
            continue
        filenames = [x for x in str(r.vintage_filename).split(";") if x]
        frames = []
        for filename in filenames:
            d = _load_ces_vintage_file(
                session,
                cache,
                refresh if first else False,
                filename,
            )
            first = False
            frames.append(_compact_ces_vintage(d, min_year, max_year))
        if not frames:
            continue
        if len(frames) == 1:
            combined = frames[0]
        else:
            combined = frames[0].set_index(["year", "month"])
            for frame in frames[1:]:
                other = frame.set_index(["year", "month"])
                common_index = combined.index.intersection(other.index)
                common_cols = combined.columns.intersection(other.columns)
                combined = (
                    combined.loc[common_index, common_cols].apply(
                        pd.to_numeric, errors="coerce"
                    )
                    + other.loc[common_index, common_cols].apply(
                        pd.to_numeric, errors="coerce"
                    )
                )
            combined = combined.reset_index()
        out[str(r.node_key)] = combined
    return out


def current_final_ces_monthly_panel(
    vintages: dict[str, pd.DataFrame],
    months: list[str],
) -> dict[str, np.ndarray]:
    out = {}
    periods = [pd.Period(m, freq="M") for m in months]
    for node_key, d in vintages.items():
        latest = d.iloc[-1]
        values = []
        ok = True
        for period in periods:
            col = f"{MONTH_ABBR[period.month]}_{period.year % 100:02d}"
            if col not in latest.index or pd.isna(latest[col]):
                ok = False
                break
            values.append(float(latest[col]) * 1000.0)
        if ok:
            out[node_key] = np.asarray(values, dtype=np.float64)
    return out


def _ces_publication_row(
    vintage: pd.DataFrame,
    cutoff: pd.Timestamp,
    monthly_release: dict[str, pd.Timestamp],
) -> tuple[pd.Series, pd.Timestamp] | None:
    eligible: list[tuple[pd.Timestamp, int]] = []
    for i, row in vintage[["year", "month"]].iterrows():
        ref = f"{int(row.year):04d}-{int(row.month):02d}"
        d = monthly_release.get(ref)
        if d is not None and d <= cutoff:
            eligible.append((d, i))
    if not eligible:
        return None
    date, idx = max(eligible, key=lambda x: x[0])
    return vintage.loc[idx], date


def _ces_month_value(row: pd.Series, month: pd.Period) -> float | None:
    col = f"{MONTH_ABBR[month.month]}_{month.year % 100:02d}"
    if col not in row.index or pd.isna(row[col]):
        return None
    return float(row[col]) * 1000.0


def _qcew_final_date(year: int, qcew_release: dict[str, pd.Timestamp]) -> pd.Timestamp | None:
    # BLS: all quarters of year Y become final when Y+1 Q1 full data are published.
    return qcew_release.get(f"{year + 1}Q1")


def build_point_in_time_history(
    *,
    session: requests.Session,
    nodes: list[dict[str, str]],
    bea_alias: dict[str, str],
    model_start: str,
    model_end: str,
    qcew_months: list[str],
    qcew_monthly: np.ndarray,
    qcew_release: dict[str, pd.Timestamp],
    ces_monthly_release: dict[str, pd.Timestamp],
    raw_bea_dir: Path,
    raw_bls_dir: Path,
    output_csv: Path,
    audit_csv: Path,
    refresh: bool,
    ces_exact_mapping: pd.DataFrame | None = None,
    ces_exact_vintages: dict[str, pd.DataFrame] | None = None,
) -> dict:
    releases = _archive_leaf_paths(session)
    snapshot_cache: dict[pd.Timestamp, dict | None] = {}
    ces_cache = raw_bls_dir / "cesvinall.zip"
    funds_vintage = _load_ces_vintage_file(session, ces_cache, False, FUNDS_VINTAGE_FILE)
    if ces_exact_mapping is None:
        ces_exact_mapping = build_ces_exact_vintage_mapping(
            session, nodes, raw_bls_dir, refresh
        )
    if ces_exact_vintages is None:
        years = [int(m[:4]) for m in qcew_months]
        ces_exact_vintages = load_exact_ces_vintages(
            session,
            ces_exact_mapping,
            raw_bls_dir,
            False,
            min(years) - 1,
            max(years),
        )
    month_index = {m: i for i, m in enumerate(qcew_months)}
    node_index = {n["key"]: i for i, n in enumerate(nodes)}
    securities_node = next(n for n in nodes if n["key"] == "securities")

    def productivity_rva_value(table: dict[str, dict], node: dict[str, str], period) -> float | None:
        value = _series_value(table, node, bea_alias, period)
        if node["key"] != "funds_trusts":
            return value
        other = _series_value(table, securities_node, bea_alias, period)
        if value is None or other is None:
            return None
        return float(value) + float(other)

    def qcew_period_value(node_key: str, period: str | int, cutoff: pd.Timestamp) -> tuple[float | None, pd.Timestamp | None]:
        if isinstance(period, int):
            year = period
            final_date = _qcew_final_date(year, qcew_release)
            months = [f"{year}-{m:02d}" for m in range(1, 13)]
        else:
            qp = pd.Period(period, freq="Q")
            year = qp.year
            final_date = _qcew_final_date(year, qcew_release)
            months = [str(m) for m in pd.period_range(qp.start_time, qp.end_time, freq="M")]
        if final_date is None or final_date > cutoff or any(m not in month_index for m in months):
            return None, final_date
        j = node_index[node_key]
        return float(np.mean([qcew_monthly[month_index[m], j] for m in months])), final_date

    def ces_period_value(
        vintage: pd.DataFrame,
        period: str | int,
        cutoff: pd.Timestamp,
    ) -> tuple[float | None, pd.Timestamp | None]:
        cache_key = (id(vintage), int(cutoff.value))
        if cache_key not in ces_row_cache:
            ces_row_cache[cache_key] = _ces_publication_row(
                vintage, cutoff, ces_monthly_release
            )
        selected = ces_row_cache[cache_key]
        if selected is None:
            return None, None
        row, release_date = selected
        if isinstance(period, int):
            months = pd.period_range(f"{period}-01", f"{period}-12", freq="M")
        else:
            qp = pd.Period(period, freq="Q")
            months = pd.period_range(qp.start_time, qp.end_time, freq="M")
        vals = [_ces_month_value(row, m) for m in months]
        if any(v is None for v in vals):
            return None, release_date
        return float(np.mean([v for v in vals if v is not None])), release_date

    def employment_period_value(
        node_key: str,
        period: str | int,
        cutoff: pd.Timestamp,
    ) -> tuple[float | None, pd.Timestamp | None, str]:
        if node_key == "funds_trusts":
            value, date = ces_period_value(funds_vintage, period, cutoff)
            if value is not None:
                return value, date, "CES_VINTAGE_523_AGG_PROXY"
        exact = ces_exact_vintages.get(node_key)
        if exact is not None:
            value, date = ces_period_value(exact, period, cutoff)
            if value is not None:
                return value, date, "CES_VINTAGE_EXACT"
        value, date = qcew_period_value(node_key, period, cutoff)
        return value, date, "QCEW_FINAL"

    ces_row_cache: dict[
        tuple[int, int], tuple[pd.Series, pd.Timestamp] | None
    ] = {}
    rows: list[dict] = []
    audit: list[dict] = []
    origins = [str(q) for q in pd.period_range(model_start, model_end, freq="Q")]

    for origin in origins:
        cutoff = pd.Period(origin, freq="Q").end_time.normalize()
        snap = _latest_snapshot(cutoff, releases, snapshot_cache, session, raw_bea_dir / "vintages", refresh)
        for node in nodes:
            key = node["key"]
            annual_tables = snap["annual"]
            qblock = snap.get("quarterly", {})

            origin_period = pd.Period(origin, freq="Q")

            def choose_latest(
                annual_candidates: list[tuple[int, object]],
                quarterly_candidates: list[tuple[str, object]],
            ) -> tuple[str, str, object] | None:
                """Choose the most recent economic period, preferring quarterly on ties."""
                best: tuple[pd.Period, int, str, str, object] | None = None
                for year, payload in annual_candidates:
                    candidate = (pd.Period(f"{year}Q4", freq="Q"), 0, "ANNUAL", str(year), payload)
                    if best is None or candidate[:2] > best[:2]:
                        best = candidate
                for quarter, payload in quarterly_candidates:
                    candidate = (pd.Period(quarter, freq="Q"), 1, "QUARTERLY", quarter, payload)
                    if best is None or candidate[:2] > best[:2]:
                        best = candidate
                if best is None:
                    return None
                return best[2], best[3], best[4]

            # Profitability uses BEA only, so do not unnecessarily wait for QCEW.
            profit_annual: list[tuple[int, tuple[float, float, str]]] = []
            annual_cva = _node_values(annual_tables.get("current_va", {}), node, bea_alias) or {}
            annual_cgo = _node_values(annual_tables.get("current_go", {}), node, bea_alias) or {}
            for year in sorted(set(annual_cva) & set(annual_cgo)):
                if pd.Period(f"{year}Q4", freq="Q") >= origin_period:
                    continue
                p_now, mode = _profitability(snap, "annual", node, bea_alias, year)
                p_prev, _ = _profitability(snap, "annual", node, bea_alias, year - 1)
                if p_now is not None and p_prev is not None:
                    profit_annual.append((int(year), (float(p_now), float(p_prev), mode)))
            profit_quarterly: list[tuple[str, tuple[float, float, str]]] = []
            if qblock:
                q_cva = _node_values(qblock.get("current_va", {}), node, bea_alias) or {}
                q_cgo = _node_values(qblock.get("current_go", {}), node, bea_alias) or {}
                for q in sorted(set(q_cva) & set(q_cgo), key=lambda x: pd.Period(x, freq="Q")):
                    qp = pd.Period(q, freq="Q")
                    prev4 = str(qp - 4)
                    if qp >= origin_period:
                        continue
                    p_now, mode = _profitability(snap, "quarterly", node, bea_alias, q)
                    p_prev, _ = _profitability(snap, "quarterly", node, bea_alias, prev4)
                    if p_now is not None and p_prev is not None:
                        profit_quarterly.append((q, (float(p_now), float(p_prev), mode)))
            profit_choice = choose_latest(profit_annual, profit_quarterly)

            # Growth also uses BEA only and can follow the latest published GO state.
            growth_annual: list[tuple[int, float]] = []
            annual_growth_table = annual_tables.get("real_go") or annual_tables.get("go_qty") or {}
            annual_growth_series = _node_values(annual_growth_table, node, bea_alias) or {}
            for year in sorted(annual_growth_series):
                if pd.Period(f"{year}Q4", freq="Q") >= origin_period or year - 1 not in annual_growth_series:
                    continue
                value = _growth_value(snap, "annual", node, bea_alias, year, year - 1)
                if value is not None:
                    growth_annual.append((int(year), float(value)))
            growth_quarterly: list[tuple[str, float]] = []
            if qblock:
                q_growth_table = qblock.get("real_go") or qblock.get("go_qty") or {}
                q_growth_series = _node_values(q_growth_table, node, bea_alias) or {}
                for q in sorted(q_growth_series, key=lambda x: pd.Period(x, freq="Q")):
                    qp = pd.Period(q, freq="Q")
                    prev4 = str(qp - 4)
                    if qp >= origin_period or prev4 not in q_growth_series:
                        continue
                    value = _growth_value(snap, "quarterly", node, bea_alias, q, prev4)
                    if value is not None:
                        growth_quarterly.append((q, float(value)))
            growth_choice = choose_latest(growth_annual, growth_quarterly)

            # Productivity requires both real VA and finalized/vintage employment.
            productivity_annual: list[
                tuple[int, tuple[float, float, float, float, pd.Timestamp, str]]
            ] = []
            annual_rva = _node_values(annual_tables.get("real_va", {}), node, bea_alias) or {}
            if key == "funds_trusts":
                sec = _node_values(annual_tables.get("real_va", {}), securities_node, bea_alias) or {}
                prod_annual_periods = set(annual_rva) & set(sec)
            else:
                prod_annual_periods = set(annual_rva)
            for year in sorted(prod_annual_periods):
                if pd.Period(f"{year}Q4", freq="Q") >= origin_period or year - 1 not in prod_annual_periods:
                    continue
                emp_now, emp_date_now, emp_source_now = employment_period_value(key, year, cutoff)
                emp_prev, emp_date_prev, emp_source_prev = employment_period_value(key, year - 1, cutoff)
                rva_now = productivity_rva_value(annual_tables["real_va"], node, year)
                rva_prev = productivity_rva_value(annual_tables["real_va"], node, year - 1)
                if (
                    emp_now is None or emp_prev is None or emp_date_now is None or emp_date_prev is None
                    or emp_now <= 0 or emp_prev <= 0 or rva_now is None or rva_prev is None
                ):
                    continue
                productivity_annual.append((
                    int(year),
                    (
                        float(rva_now), float(rva_prev), float(emp_now), float(emp_prev),
                        max(emp_date_now, emp_date_prev),
                        emp_source_now if emp_source_now == emp_source_prev else f"{emp_source_now}+{emp_source_prev}",
                    ),
                ))
            productivity_quarterly: list[
                tuple[str, tuple[float, float, float, float, pd.Timestamp, str]]
            ] = []
            if qblock:
                q_rva = _node_values(qblock.get("real_va", {}), node, bea_alias) or {}
                if key == "funds_trusts":
                    q_sec = _node_values(qblock.get("real_va", {}), securities_node, bea_alias) or {}
                    prod_q_periods = set(q_rva) & set(q_sec)
                else:
                    prod_q_periods = set(q_rva)
                for q in sorted(prod_q_periods, key=lambda x: pd.Period(x, freq="Q")):
                    qp = pd.Period(q, freq="Q")
                    prev4 = str(qp - 4)
                    if qp >= origin_period or prev4 not in prod_q_periods:
                        continue
                    emp_now, emp_date_now, emp_source_now = employment_period_value(key, q, cutoff)
                    emp_prev, emp_date_prev, emp_source_prev = employment_period_value(key, prev4, cutoff)
                    rva_now = productivity_rva_value(qblock["real_va"], node, q)
                    rva_prev = productivity_rva_value(qblock["real_va"], node, prev4)
                    if (
                        emp_now is None or emp_prev is None or emp_date_now is None or emp_date_prev is None
                        or emp_now <= 0 or emp_prev <= 0 or rva_now is None or rva_prev is None
                    ):
                        continue
                    productivity_quarterly.append((
                        q,
                        (
                            float(rva_now), float(rva_prev), float(emp_now), float(emp_prev),
                            max(emp_date_now, emp_date_prev),
                            emp_source_now if emp_source_now == emp_source_prev else f"{emp_source_now}+{emp_source_prev}",
                        ),
                    ))
            productivity_choice = choose_latest(productivity_annual, productivity_quarterly)

            if profit_choice is None or growth_choice is None or productivity_choice is None:
                raise RuntimeError(
                    f"Incomplete component-specific PIT state for {origin}/{key}: "
                    f"P={profit_choice is not None} Prod={productivity_choice is not None} G={growth_choice is not None}"
                )

            profitability_resolution, profitability_source, profit_payload = profit_choice
            productivity_resolution, productivity_source, prod_payload = productivity_choice
            growth_resolution, growth_source, growth = growth_choice
            profit_now, profit_prev, gos_mode = profit_payload
            rva, rva_prev, employment, employment_prev, emp_date, employment_source = prod_payload
            profitability_momentum = float(profit_now) - float(profit_prev)
            productivity_now = float(rva) / float(employment)
            productivity_prev = float(rva_prev) / float(employment_prev)
            productivity_scale = abs(productivity_now) + abs(productivity_prev)
            if productivity_scale <= 1e-15:
                raise RuntimeError(f"Degenerate productivity change for {origin}/{key}")
            # Symmetric percentage change is invariant to BEA chained-dollar
            # rebasing and remains finite when real VA is negative or crosses zero.
            productivity_growth = (
                2.0 * (productivity_now - productivity_prev) / productivity_scale
            )
            rows.append({
                "quarter": origin,
                "node_key": key,
                "node_name": node["name"],
                "history_profitability_raw": profitability_momentum,
                "history_productivity_raw": productivity_growth,
                "history_growth_raw": float(growth),
            })
            audit.append({
                "quarter": origin,
                "information_cutoff_date": cutoff.date().isoformat(),
                "node_key": key,
                "node_name": node["name"],
                # Legacy aliases retain the employment-limited productivity state.
                "state_resolution": productivity_resolution,
                "state_source_period": productivity_source,
                "profitability_resolution": profitability_resolution,
                "profitability_source_period": profitability_source,
                "productivity_resolution": productivity_resolution,
                "productivity_source_period": productivity_source,
                "growth_resolution": growth_resolution,
                "growth_source_period": growth_source,
                "bea_vintage_release_date": snap["release_date"].date().isoformat(),
                "bea_archive_path": snap["archive_path"],
                "employment_source": employment_source,
                "employment_available_date": emp_date.date().isoformat(),
                "gos_mode": gos_mode,
                "history_profitability_raw": profitability_momentum,
                "history_productivity_raw": productivity_growth,
                "history_growth_raw": float(growth),
            })

    df = pd.DataFrame(rows)
    adf = pd.DataFrame(audit)
    expected = len(origins) * len(nodes)
    if len(df) != expected or df.duplicated(["quarter", "node_key"]).any():
        raise RuntimeError(f"Invalid point-in-time history grid: {len(df)} != {expected}")
    if not np.isfinite(df[["history_profitability_raw", "history_productivity_raw", "history_growth_raw"]].to_numpy()).all():
        raise RuntimeError("Non-finite point-in-time history")
    if (pd.to_datetime(adf["bea_vintage_release_date"]) > pd.to_datetime(adf["information_cutoff_date"])).any():
        raise RuntimeError("BEA vintage release exceeds cutoff")
    if (pd.to_datetime(adf["employment_available_date"]) > pd.to_datetime(adf["information_cutoff_date"])).any():
        raise RuntimeError("Employment vintage/finalization exceeds cutoff")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    audit_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    adf.to_csv(audit_csv, index=False, encoding="utf-8-sig")
    return {
        "rows": len(df),
        "period": [origins[0], origins[-1]],
        "nodes": len(nodes),
        "bea_vintages_used": int(adf["bea_vintage_release_date"].nunique()),
        "annual_rows": int((adf["state_resolution"] == "ANNUAL").sum()),
        "quarterly_rows": int((adf["state_resolution"] == "QUARTERLY").sum()),
        "component_resolution_rows": {
            component: {
                "annual": int((adf[f"{component}_resolution"] == "ANNUAL").sum()),
                "quarterly": int((adf[f"{component}_resolution"] == "QUARTERLY").sum()),
            }
            for component in ("profitability", "productivity", "growth")
        },
        "parent_gos_proxy_rows": int(adf["gos_mode"].str.startswith("PARENT_SHARE:").sum()),
        "employment_sources": adf["employment_source"].value_counts().to_dict(),
        "component_definition": {
            "profitability": "4Q/year-over-year change in GOS-to-gross-output margin",
            "productivity": "4Q/year-over-year symmetric change in real value added per worker; scale-invariant to chained-dollar rebasing and safe across sign changes",
            "growth": "4Q/year-over-year real gross-output growth",
        },
        "revision_policy": (
            "BEA official archive vintage; exact CES publication-vintage employment where available; "
            "QCEW current-final values only after official finalization as fallback; "
            "funds/trusts continuous CES 523+525 vintage proxy; each component uses its freshest "
            "independently available source period"
        ),
        "status": "PASS",
    }

def main() -> None:
    parser = argparse.ArgumentParser(description="Collect all official PRISM target-construction inputs")
    parser.add_argument("--refresh", action="store_true", help="redownload official source data even if cached")
    args = parser.parse_args()
    collect(args.refresh)


if __name__ == "__main__":
    main()
