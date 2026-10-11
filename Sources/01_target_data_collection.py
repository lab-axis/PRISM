#!/usr/bin/env python3
"""Collect the latest official source levels for PRISM's 56 industries.

No retrospective publication-vintage reconstruction, no target/ICI calculation.
BEA: latest industry-account release. CES: latest public historical employment
estimate. QCEW: current monthly private-industry employment. Raw source units
and original reference periods are retained without interpolation.
"""
from __future__ import annotations
import csv
import hashlib
import io
import json
import math
import re
import zipfile
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path, PurePosixPath
from urllib.parse import quote
import numpy as np
import pandas as pd
import requests
import xlrd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Data/01_target_data_collection"
RESULTS = ROOT / "Results/01_target_data_collection"
QCEW_SNAPSHOT = DATA / "raw_latest/qcew_monthly_employment_2003_2025.json"
QCEW_MANIFEST = DATA / "raw_latest/source_manifest.json"
BEA_API = "https://apps.bea.gov/histdata/core/data"
BEA_FILES = "https://apps.bea.gov/HistData/Files/"
CES_VINTAGES = "https://www.bls.gov/web/empsit/cesvinall.zip"
FUNDS_CES_ID = "CEU5552300001"
FUNDS_CES_MEMBER = "tri_555230_NSA.csv"
FUNDS_CES_MEASURE = "employment_523_525_aggregate"
QCEW_API = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
AS_OF = date.today()
YEAR_START, YEAR_END = 2003, 2025


NODES = """oil_gas_extraction|Oil and gas extraction|102110
mining_except_oil_gas|Mining, except oil and gas|102120
support_mining|Support activities for mining|102130
utilities|Utilities|442200
construction|Construction|200000
wood_products|Wood products|313210
nonmetallic_mineral|Nonmetallic mineral products|313270
primary_metals|Primary metals|313310
fabricated_metal|Fabricated metal products|313320
machinery|Machinery|313330
computer_electronic|Computer and electronic products|313340
electrical_equipment|Electrical equipment, appliances, and components|313350
motor_vehicles|Motor vehicles, bodies and trailers, and parts|313360
other_transport_equipment|Other transportation equipment|
furniture|Furniture and related products|313370
misc_manufacturing|Miscellaneous manufacturing|313390
food_beverage_tobacco|Food and beverage and tobacco products|
textiles|Textile mills and textile product mills|323130;323140
apparel_leather|Apparel and leather and allied products|
paper|Paper products|323220
printing|Printing and related support activities|323230
petroleum_coal|Petroleum and coal products|323240
chemicals|Chemical products|323250
plastics_rubber|Plastics and rubber products|323260
wholesale|Wholesale trade|414200
retail|Retail trade|420000
air_transport|Air transportation|434810
rail_transport|Rail transportation|434820
water_transport|Water transportation|434830
truck_transport|Truck transportation|434840
transit_ground|Transit and ground passenger transportation|434850
pipeline|Pipeline transportation|434860
other_transport_support|Other transportation and support activities|434870;434880
warehousing|Warehousing and storage|434930
information|Information|500000
credit_intermediation|Federal Reserve banks, credit intermediation, and related activities|555210;555220
securities|Securities, commodity contracts, and investments|
insurance|Insurance carriers and related activities|555240
funds_trusts|Funds, trusts, and other financial vehicles|555230
real_estate|Real estate|555310
rental_leasing|Rental and leasing services and lessors of intangible assets|555320;555330
legal|Legal services|
computer_systems|Computer systems design and related services|
misc_professional|Miscellaneous professional, scientific, and technical services|
management_companies|Management of companies and enterprises|605500
admin_support|Administrative and support services|605610
waste_remediation|Waste management and remediation services|605620
education|Educational services|656100
ambulatory_health|Ambulatory health care services|656210
hospitals_nursing|Hospitals, nursing and residential care facilities|656220;656230
social_assistance|Social assistance|656240
performing_arts_museums|Performing arts, spectator sports, museums, and related activities|707110;707120
amusement_recreation|Amusements, gambling, and recreation industries|707130
accommodation|Accommodation|707210
food_services|Food services and drinking places|707220
other_services|Other services, except government|800000"""
INDUSTRIES = [tuple(row.split("|")) for row in NODES.splitlines()]
MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()

NAICS = """211|212|213|22|23|321|327|331|332|333|334|335|3361,3362,3363|3364,3365,3366,3369|337|339|311,312|313,314|315,316|322|323|324|325|326|42|44-45|481|482|483|484|485|486|487,488|493|51|521,522|523|524|525|531|532,533|5411|5415|5412,5413,5414,5416,5417,5418,5419|55|561|562|61|621|622,623|624|711,712|713|721|722|81""".split("|")
assert len(NAICS) == len(INDUSTRIES) == 56
NODE_NAICS = dict(zip((n[0] for n in INDUSTRIES), NAICS))

def get(session: requests.Session, url: str) -> bytes:
    response = session.get(url, timeout=120)
    response.raise_for_status()
    return response.content

def release_date_from_path(path: str) -> date | None:
    match = re.search(r"(January|February|March|April|May|June|July|August|September|October|November|December)-"
                      r"(\d{1,2})-(\d{4})$", path, re.I)
    if not match:
        return None
    from datetime import datetime
    return datetime.strptime(" ".join(match.groups()), "%B %d %Y").date()

def bea_release_paths(session: requests.Session) -> list[tuple[date, list[str]]]:
    response = session.get(BEA_API + "/Fea_DisplayChildrenC/",
                           params={"HistMainId": 8, "getFiles": "false", "getDirs": "true"}, timeout=45)
    response.raise_for_status()
    grouped: dict[date, list[str]] = {}
    for path in response.json().get("FileArray", []):
        if "GDP_by_Industry" not in path or path.endswith("_notes"):
            continue
        published = release_date_from_path(path)
        if published and published <= AS_OF:
            grouped.setdefault(published, []).append(path)
    return sorted(grouped.items())

def bea_archive_files(session: requests.Session, path: str) -> list[str]:
    response = session.get(BEA_API + "/UrlPath_getID/", params={"UrlPath": path}, timeout=45)
    response.raise_for_status()
    ids = response.json()
    if not ids:
        return []
    response = session.get(BEA_API + "/getPath/" + str(ids[0]["Theid"]), timeout=45)
    response.raise_for_status()
    resolved = response.json()
    if not resolved:
        return []
    response = session.get(BEA_API + "/Fea_DisplayChildrenC/",
                           params={"HistMainId": 8, "thePath": resolved[0]["Thepath"],
                                   "getFiles": "true", "getDirs": "false"}, timeout=45)
    response.raise_for_status()
    return response.json().get("Filearray3", []) or []

def bea_url(archive_path: str) -> str:
    posix = archive_path.replace("\\", "/")
    prefix = "/Inetpub/wwwroot/website/website/HistData/Files/"
    if not posix.startswith(prefix):
        raise ValueError("Unrecognized BEA archive path")
    return BEA_FILES + quote(posix[len(prefix):], safe="/")

def excel_rows(data: bytes, filename: str, sheet: str) -> list[tuple]:
    if filename.casefold().endswith(".xlsx"):
        # Parse workbook XML directly so the collector does not depend on a
        # newer XLSX package or persist complete (revision-bearing) workbooks.
        ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            wb = ET.fromstring(archive.read("xl/workbook.xml"))
            rel = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
            relations = {r.attrib["Id"]: r.attrib["Target"] for r in rel}
            shared = []
            if "xl/sharedStrings.xml" in archive.namelist():
                strings = ET.fromstring(archive.read("xl/sharedStrings.xml"))
                shared = ["".join(t.text or "" for t in s.iterfind(".//m:t", ns))
                          for s in strings.findall("m:si", ns)]
            sheet_path = None
            for item in wb.findall("m:sheets/m:sheet", ns):
                if item.attrib.get("name") == sheet:
                    rel_id = item.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
                    target = relations[rel_id]
                    sheet_path = target.lstrip("/") if target.startswith("/") else str(PurePosixPath("xl") / target)
                    break
            if sheet_path is None:
                raise KeyError(sheet)
            document = ET.fromstring(archive.read(sheet_path))
            cells = []
            max_col = 0
            for row in document.findall(".//m:sheetData/m:row", ns):
                values = {}
                for cell in row.findall("m:c", ns):
                    ref = re.match(r"[A-Z]+", cell.attrib.get("r", "A1"))
                    index = 0
                    for c in ref.group():
                        index = index * 26 + ord(c) - 64
                    index -= 1
                    max_col = max(max_col, index)
                    v = cell.find("m:v", ns)
                    inline = cell.find("m:is", ns)
                    if cell.attrib.get("t") == "s" and v is not None:
                        value = shared[int(v.text)]
                    elif inline is not None:
                        value = "".join(t.text or "" for t in inline.iterfind(".//m:t", ns))
                    else:
                        value = v.text if v is not None else ""
                    values[index] = value
                cells.append(values)
            return [tuple(row.get(i, "") for i in range(max_col + 1)) for row in cells]
    book = xlrd.open_workbook(file_contents=data, on_demand=True)
    try:
        ws = book.sheet_by_name(sheet)
        return [tuple(ws.row_values(i)) for i in range(ws.nrows)]
    finally:
        book.release_resources()

def workbook_sheet(data: bytes, filename: str, *candidates: str) -> list[tuple]:
    for name in candidates:
        try:
            return excel_rows(data, filename, name)
        except KeyError:
            continue
        except xlrd.biffh.XLRDError as exc:
            if "No sheet named" not in str(exc):
                raise
    raise KeyError("Workbook has none of: " + ", ".join(candidates))

def number(value) -> float | None:
    try:
        x = float(str(value).replace(",", "").strip())
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None

def name(value) -> str:
    label = re.sub(r"\s+", " ", str(value or "").replace("…", " ")).strip()
    return re.sub(r"\s*(?:[/\\]+\d+[/\\]+|\[\d+\])\s*$", "", label).strip()

def extract_table(rows: list[tuple], frequency: str, *, gos: bool = False) -> tuple[dict, str]:
    """Extract source cells, preserving their values and original units."""
    raw_header = " ".join(str(v) for row in rows[:8] for v in row[:5]).lower()
    # Preserve original scale AND chained-dollar reference year; do not mix
    # rebased real series or annualized/non-annualized quantities downstream.
    unit = " | ".join(str(v).strip() for row in rows[:5] for v in row
                      if v and ("[" in str(v) or "dollar" in str(v).lower()
                                or "annual rates" in str(v).lower()))
    if not unit:
        raise DataIntegrityError("Source unit/base year is absent from the table header")
    periods: list[tuple[int, str]] = []
    header_index = -1
    for i, row in enumerate(rows):
        if frequency == "annual":
            found = [(j, str(int(float(v)))) for j, v in enumerate(row)
                     if re.fullmatch(r"(?:19|20)\d{2}(?:\.0)?", str(v).strip())]
        else:
            found = [(j, str(v).strip().upper()) for j, v in enumerate(row)
                     if re.fullmatch(r"(?:19|20)\d{2}Q[1-4]", str(v).strip().upper())]
        if len(found) >= 2:
            periods, header_index = found, i
            break
    if header_index < 0 and frequency == "quarterly":
        for i, row in enumerate(rows):
            roman = {"I": 1, "II": 2, "III": 3, "IV": 4}
            if sum(str(v).strip().upper() in roman for v in row) < 4 or i == 0:
                continue
            last_year = None
            for j, v in enumerate(rows[i - 1]):
                if re.fullmatch(r"(?:19|20)\d{2}(?:\.0)?", str(v).strip()):
                    last_year = int(float(v))
                if j < len(row) and str(row[j]).strip().upper() in roman and last_year:
                    periods.append((j, f"{last_year}Q{roman[str(row[j]).strip().upper()]}"))
            if periods:
                header_index = i
                break
    if header_index < 0:
        raise ValueError("No " + frequency + " header found")
    output: dict[tuple[str, str], float] = {}
    parent = ""
    for row in rows[header_index + 1:]:
        if len(row) < 2:
            continue
        label = name(row[1])
        if not label:
            continue
        if gos:
            low = label.lower()
            if low == "gross operating surplus":
                label = parent
            elif low in {"compensation of employees", "taxes on production and imports less subsidies"}:
                continue
            else:
                parent = label
                continue
            if not label:
                continue
        for column, period in periods:
            if column >= len(row):
                continue
            value = number(row[column])
            if value is not None:
                output[(label, period)] = value
    return output, unit

def inspect_bea_package(blob: bytes, outer_filename: str) -> list[tuple[str, str, str, str, float, str]]:
    """Return (series, frequency, original industry, period, value, units)."""
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        files = {Path(member).name.lower(): member for member in archive.namelist()}
        output: list[tuple[str, str, str, str, float, str]] = []
        def source_file(stem: str):
            return next(((files[stem + suffix], stem + suffix) for suffix in (".xlsx", ".xls")
                         if stem + suffix in files), None)

        quarterly_only = "qtr" in outer_filename.lower()
        va_file = source_file("valueaddedqtr" if quarterly_only else "valueadded")
        go_file = source_file("grossoutputqtr" if quarterly_only else "grossoutput")
        if va_file is None or go_file is None:
            raise ValueError("Unrecognized BEA archive workbook names: " + outer_filename)
        va_name, go_name = va_file[0], go_file[0]
        va, go = archive.read(va_name), archive.read(go_name)
        specs = []
        if not quarterly_only:
            specs += [
                ("current_va", "annual", va, va_name, ("TVA105-A", "VA", "Cu$"), False),
                ("real_va", "annual", va, va_name, ("TVA106-A", "RealVA"), False),
                ("gos", "annual", va, va_name, ("TVA113-A", "Components", "ComponentsCu$"), True),
                ("current_go", "annual", go, go_name, ("TGO105-A", "GO", "CurrentDollars"), False),
                ("real_go", "annual", go, go_name, ("TGO106-A", "Real GO", "RGO"), False),
                ("go_qty", "annual", go, go_name, ("ChainQtyIndexes",), False),
            ]
        specs += [
            ("current_va", "quarterly", va, va_name, ("TVA105-Q", "VA") if quarterly_only else ("TVA105-Q",), False),
            ("real_va", "quarterly", va, va_name, ("TVA106-Q", "RealVA") if quarterly_only else ("TVA106-Q",), False),
            ("current_go", "quarterly", go, go_name, ("TGO105-Q", "GO") if quarterly_only else ("TGO105-Q",), False),
            ("real_go", "quarterly", go, go_name, ("TGO106-Q", "Real GO", "RGO") if quarterly_only else ("TGO106-Q",), False),
        ]
        if quarterly_only:
            specs.append(("go_qty", "quarterly", go, go_name, ("ChainQtyIndexes",), False))
        for series, freq, data, fname, sheets, gos in specs:
            try:
                matrix = workbook_sheet(data, fname, *sheets)
                # Legacy annual archives have VA/GO tabs too: a quarterly
                # header must be present before these can be quarterly data.
                entries, units = extract_table(matrix, freq, gos=gos)
            except KeyError:
                continue
            output.extend((series, freq, industry, period, value, units)
                          for (industry, period), value in entries.items())
        employment_name = next((files[n] for n in ("employment.xlsx", "employment.xls") if n in files), None)
        if employment_name and not quarterly_only:
            employment_data = archive.read(employment_name)
            for series, sheet in (
                ("employees_full_part_time", "Full-Time and Part-Time"),
                ("employment_full_time_equivalent", "Full-Time Equivalent"),
            ):
                try:
                    matrix = workbook_sheet(employment_data, employment_name, sheet)
                    entries, units = extract_table(matrix, "annual")
                except KeyError:
                    continue
                output.extend((series, "annual", industry, period, value, "thousands_of_people")
                              for (industry, period), value in entries.items())
    if not output:
        raise ValueError("No BEA observations found")
    return output

BEA_REQUIRED = {
    "annual": ("current_va", "real_va", "current_go", "real_go", "gos"),
    "quarterly": ("current_va", "real_va", "current_go", "real_go"),
}
MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()


def collect_latest_bea(session: requests.Session) -> tuple[list[dict], dict]:
    """Find the newest *supported* BEA release, never replay older vintages."""
    releases = bea_release_paths(session)
    for published, paths in reversed(releases):
        for path in paths:
            files = bea_archive_files(session, path)
            candidates = [
                candidate for candidate in files
                if candidate.replace("\\", "/").split("/")[-1].casefold() == "gdpbyind.zip"
            ]
            if not candidates:
                continue
            url = bea_url(candidates[0])
            try:
                observations = inspect_bea_package(get(session, url), "GdpByInd.zip")
            except (ValueError, KeyError, zipfile.BadZipFile):
                continue
            rows = []
            for series, freq, industry, period, value, units in observations:
                if not (YEAR_START <= int(period[:4]) <= YEAR_END):
                    continue
                if series not in BEA_REQUIRED.get(freq, ()):
                    continue
                rows.append({
                    "provider": "BEA", "series": series, "frequency": freq,
                    "period": period, "source_industry": industry,
                    "value": value, "units": units,
                    "source_url": url, "source_release_date": published.isoformat(),
                })
            if rows:
                return rows, {"archive": url, "release_date": published.isoformat(),
                              "native_observations": len(rows)}
    raise RuntimeError("No supported current BEA GDP-by-industry release was found")


def collect_latest_rail_ces(session: requests.Session) -> tuple[list[dict], dict]:
    """Use the most recently available actual vintage ROW, not vintage replay."""
    source = "tri_434820_NSA.csv"
    with zipfile.ZipFile(io.BytesIO(get(session, CES_VINTAGES))) as archive:
        table = pd.read_csv(io.BytesIO(archive.read(source)))
    table = table.sort_values(["year", "month"])
    table = table[table["year"].astype(int) * 100 + table["month"].astype(int)
                  <= AS_OF.year * 100 + AS_OF.month]
    if table.empty:
        raise RuntimeError("CES rail series has no current public observation")
    latest = table.iloc[-1]
    rows = []
    for column in table.columns:
        match = re.fullmatch(r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)_(\d{2})", column)
        if not match:
            continue
        month = MONTHS.index(match.group(1)) + 1
        year = 2000 + int(match.group(2))
        if not YEAR_START <= year <= YEAR_END:
            continue
        value = number(latest[column])
        if value is None or value < 0:
            continue
        rows.append({
            "provider": "BLS_CES", "series": "employment",
            "frequency": "monthly", "period": f"{year:04d}-{month:02d}",
            "source_industry": source, "value": value,
            "units": "thousands_of_people",
            "source_url": CES_VINTAGES + "#" + source,
            "source_release_date": f"{int(latest['year']):04d}-{int(latest['month']):02d}",
        })
    return rows, {"archive": CES_VINTAGES, "source_file": source,
                  "latest_vintage_month": f"{int(latest['year']):04d}-{int(latest['month']):02d}",
                  "native_observations": len(rows)}


def collect_latest_funds_ces(session: requests.Session) -> tuple[list[dict], dict]:
    """Collect the published 523+525 CES employment scope used by productivity.

    Standalone QCEW NAICS 525 does not provide a stable denominator around the
    2013 securities/funds industry reclassification. Preserve the combined
    original CES level as a separate native source; never overwrite QCEW 525.
    Prefer FRED's current republication of the exact BLS NSA series. The
    latest BLS CES archive row is a fallback when FRED is unavailable.
    """
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    source_url = url + "?id=" + FUNDS_CES_ID
    try:
        # The public FRED CSV export can reject browser user agents. Do not
        # inherit the BLS/BEA session headers for this specific endpoint.
        response = requests.get(url, params={"id": FUNDS_CES_ID}, timeout=(6, 20))
        response.raise_for_status()
        table = pd.read_csv(io.BytesIO(response.content))
        date_col = "observation_date" if "observation_date" in table else "DATE"
        if date_col not in table or FUNDS_CES_ID not in table:
            raise ValueError("Invalid current CES 523+525 series format")
        dates = pd.to_datetime(table[date_col], errors="raise")
        if not dates.dt.day.eq(1).all():
            raise ValueError("CES 523+525 source must have monthly reference dates")
        monthly = pd.DataFrame({
            "period": dates.dt.to_period("M").astype(str),
            "value": pd.to_numeric(table[FUNDS_CES_ID], errors="coerce"),
        })
        origin = "current_FRED_BLS_series"
        release_marker = ""
        source_url = response.url
    except (requests.RequestException, ValueError, KeyError) as exc:
        # Same native industry source, using the latest *row* of the official
        # BLS revision archive, never reconstructing historical vintages.
        with zipfile.ZipFile(io.BytesIO(get(session, CES_VINTAGES))) as archive:
            table = pd.read_csv(io.BytesIO(archive.read(FUNDS_CES_MEMBER)))
        table = table.sort_values(["year", "month"])
        table = table[table["year"].astype(int) * 100 + table["month"].astype(int)
                      <= AS_OF.year * 100 + AS_OF.month]
        if table.empty:
            raise RuntimeError("No current combined securities/funds CES release") from exc
        latest = table.iloc[-1]
        values = []
        for column in table.columns:
            match = re.fullmatch(
                r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)_(\d{2})", column)
            if match:
                year = 2000 + int(match.group(2))
                month = MONTHS.index(match.group(1)) + 1
                values.append((f"{year:04d}-{month:02d}", number(latest[column])))
        monthly = pd.DataFrame(values, columns=["period", "value"])
        origin = "latest_BLS_CES_vintage_row"
        release_marker = f"{int(latest['year']):04d}-{int(latest['month']):02d}"
        source_url = CES_VINTAGES + "#" + FUNDS_CES_MEMBER

    monthly = monthly[monthly.period.str[:4].astype(int).between(YEAR_START, YEAR_END)]
    expected = pd.period_range(f"{YEAR_START}-01", f"{YEAR_END}-12", freq="M").astype(str)
    monthly = monthly.set_index("period").reindex(expected)
    if (monthly.value.isna().any() or (monthly.value <= 0).any()
            or not np.isfinite(monthly.value.to_numpy(float)).all()):
        raise RuntimeError("CES 523+525 combined employment is incomplete")
    rows = [
        {"provider": "BLS_CES", "series": FUNDS_CES_MEASURE,
         "frequency": "monthly", "period": month,
         "source_industry": FUNDS_CES_ID, "value": float(val),
         "units": "thousands_of_people", "source_url": source_url,
         "source_release_date": release_marker}
        for month, val in monthly.value.items()
    ]
    return rows, {"series_id": FUNDS_CES_ID, "source_scope": "NAICS_523_PLUS_525",
                  "source_url": source_url, "origin": origin, "months": len(rows)}


def load_verified_qcew_snapshot(codes: list[str]) -> tuple[dict, str]:
    """Use an explicitly preserved source artifact, never a Git HEAD lookup.

    The source manifest pins its SHA-256 and official BLS API provenance.
    If there is no local snapshot and the API is throttled, fail visibly.
    """
    if not QCEW_SNAPSHOT.is_file() or not QCEW_MANIFEST.is_file():
        raise RuntimeError(
            "BLS QCEW API unavailable and portable official QCEW snapshot/manifest "
            "not found under Data/01_target_data_collection/raw_latest"
        )
    blob = QCEW_SNAPSHOT.read_bytes()
    manifest = json.loads(QCEW_MANIFEST.read_text(encoding="utf-8"))
    relative = QCEW_SNAPSHOT.relative_to(ROOT).as_posix()
    provenance = manifest.get("files", {}).get(relative)
    digest = hashlib.sha256(blob).hexdigest()
    if not provenance or provenance.get("sha256") != digest:
        raise RuntimeError("QCEW snapshot provenance or SHA-256 integrity check failed")
    source = json.loads(blob)
    if set(source) != set(codes):
        raise RuntimeError("QCEW snapshot code coverage disagrees with the 56-node registry")
    expected_months = set(pd.period_range(
        f"{YEAR_START}-01", f"{YEAR_END}-12", freq="M").astype(str))
    for code in codes:
        monthly = source[code]
        if (set(monthly) != expected_months
                or any(number(value) is None or float(value) < 0
                       for value in monthly.values())):
            raise RuntimeError(f"QCEW snapshot contains incomplete/invalid native data: {code}")
    return source, digest


def save_qcew_snapshot(rows: list[dict], codes: list[str]) -> None:
    """Persist a complete successful API result as a portable official source."""
    data = {code: {} for code in codes}
    for row in rows:
        monthly = data[row["source_industry"]]
        if row["period"] in monthly:
            raise RuntimeError("Duplicate monthly employment in live QCEW API data")
        monthly[row["period"]] = row["value"]
    expected_months = set(pd.period_range(
        f"{YEAR_START}-01", f"{YEAR_END}-12", freq="M").astype(str))
    if any(set(data[code]) != expected_months for code in codes):
        raise RuntimeError("Live QCEW API data are not complete; cache not replaced")
    QCEW_SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    blob = (json.dumps(data, indent=2, sort_keys=True) + "\n").encode("utf-8")
    digest = hashlib.sha256(blob).hexdigest()
    manifest = {
        "source_type": "official_BLS_data_snapshot",
        "runtime_git_dependency": False,
        "files": {
            QCEW_SNAPSHOT.relative_to(ROOT).as_posix(): {
                "sha256": digest, "bytes": len(blob),
                "source_url": QCEW_API, "kind": "QCEW API industry employment",
                "origin": "complete_live_BLS_QCEW_API_result",
            }
        },
    }
    # Write a complete verified snapshot; do not rely on deleted project files.
    QCEW_SNAPSHOT.write_bytes(blob)
    QCEW_MANIFEST.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def collect_latest_qcew(session: requests.Session) -> tuple[list[dict], dict]:
    """Fetch the BLS current-data API with the exact source NAICS codes."""
    codes = sorted({code for spec in NODE_NAICS.values() for code in spec.split(",")})
    rows: list[dict] = []
    calls = 0
    try:
        for start in range(YEAR_START, YEAR_END + 1, 10):
            end = min(start + 9, YEAR_END)
            for offset in range(0, len(codes), 20):
                ids = {"ENUUS000105" + code: code for code in codes[offset:offset + 20]}
                response = session.post(
                    QCEW_API, json={"seriesid": list(ids), "startyear": str(start),
                                    "endyear": str(end)}, timeout=70,
                )
                response.raise_for_status()
                data = response.json()
                if data.get("status") != "REQUEST_SUCCEEDED":
                    raise RuntimeError(f"BLS current QCEW API failed: {data.get('message')}")
                sources = {item["seriesID"]: item.get("data", [])
                           for item in data.get("Results", {}).get("series", [])}
                for sid, naics in ids.items():
                    if not sources.get(sid):
                        raise RuntimeError(f"BLS current QCEW returned no source: {sid}/{start}-{end}")
                    for entry in sources[sid]:
                        if not re.fullmatch(r"M(?:0[1-9]|1[0-2])", entry.get("period", "")):
                            continue
                        year = int(entry["year"])
                        if not YEAR_START <= year <= YEAR_END:
                            continue
                        val = number(entry.get("value"))
                        if val is None or val < 0:
                            continue
                        rows.append({
                            "provider": "BLS_QCEW", "series": "employment",
                            "frequency": "monthly",
                            "period": f"{year:04d}-{int(entry['period'][1:]):02d}",
                            "source_industry": naics, "value": val, "units": "persons",
                            "source_url": f"{QCEW_API}{sid}?startyear={start}&endyear={end}",
                            "source_release_date": "",
                        })
                calls += 1
        save_qcew_snapshot(rows, codes)
        return rows, {"api": QCEW_API, "calls": calls, "naics_codes": len(codes),
                      "native_observations": len(rows), "origin": "live_current_api"}
    except (requests.RequestException, RuntimeError) as exc:
        # A throttled API is not a reason to restore deleted files from HEAD.
        # An immutable, SHA-pinned official source snapshot is a valid input;
        # incomplete/missing snapshots are hard errors, not fabricated data.
        source, digest = load_verified_qcew_snapshot(codes)
        rows = [
            {"provider": "BLS_QCEW", "series": "employment", "frequency": "monthly",
             "period": reference, "source_industry": naics,
             "value": float(value), "units": "persons",
             "source_url": QCEW_API + "ENUUS000105" + naics,
             "source_release_date": ""}
            for naics in codes for reference, value in source[naics].items()
            if YEAR_START <= int(reference[:4]) <= YEAR_END and number(value) is not None
        ]
        return rows, {"api": QCEW_API, "naics_codes": len(codes),
                      "native_observations": len(rows),
                      "origin": "verified_portable_official_source_snapshot",
                      "snapshot_path": QCEW_SNAPSHOT.relative_to(ROOT).as_posix(),
                      "snapshot_sha256": digest,
                      "live_api_issue": str(exc)[:200]}


def assign_nodes(bea: list[dict], rail: list[dict], qcew: list[dict],
                 funds_aggregate: list[dict] | None = None) -> pd.DataFrame:
    """Keep original native observations, and exact component identities."""
    index = {}
    for node, industry, _ in INDUSTRIES:
        components = (["Hospitals", "Nursing and residential care facilities"]
                      if node == "hospitals_nursing" else [industry])
        for component in components:
            index.setdefault(name(component).casefold(), []).append(node)
    rows = []
    for raw in bea:
        for node in index.get(name(raw["source_industry"]).casefold(), []):
            rows.append({**raw, "node_key": node, "component": raw["source_industry"]})
    for raw in rail:
        rows.append({**raw, "node_key": "rail_transport", "component": raw["source_industry"]})
    for raw in funds_aggregate or []:
        if (raw["provider"] != "BLS_CES" or raw["series"] != FUNDS_CES_MEASURE
                or raw["source_industry"] != FUNDS_CES_ID):
            raise RuntimeError("Unexpected CES industry scope in funds aggregate")
        rows.append({**raw, "node_key": "funds_trusts", "component": FUNDS_CES_ID})
    lookup = {naics: node for node, codes in NODE_NAICS.items()
              if node != "rail_transport" for naics in codes.split(",")}
    if len(lookup) != sum(len(codes.split(",")) for node, codes in NODE_NAICS.items()
                          if node != "rail_transport"):
        raise RuntimeError("Overlapping QCEW NAICS component mapping: would duplicate employment")
    for raw in qcew:
        node = lookup.get(raw["source_industry"])
        if node:
            rows.append({**raw, "node_key": node, "component": raw["source_industry"]})
    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError("No latest source records matched the PRISM industries")
    key = ["provider", "series", "frequency", "period", "node_key", "component"]
    if result.duplicated(key).any():
        raise RuntimeError("Latest source contains duplicate native observations")
    result["year"] = result["period"].str[:4].astype(int)
    return result.sort_values(key, kind="stable").reset_index(drop=True)


def coverage_audit(data: pd.DataFrame) -> pd.DataFrame:
    """Verify native official levels; no interpolation, annual-to-quarterly mixing."""
    expected = []
    for node, industry, _ in INDUSTRIES:
        parts = (["Hospitals", "Nursing and residential care facilities"]
                 if node == "hospitals_nursing" else [industry])
        for series in ("current_va", "real_va", "current_go", "real_go", "gos"):
            for year in range(2005, YEAR_END + 1):
                for part in parts:
                    expected.append((node, series, "annual", str(year), part))
        for series in ("current_va", "real_va", "current_go", "real_go"):
            for year in range(2005, YEAR_END + 1):
                for quarter in range(1, 5):
                    for part in parts:
                        expected.append((node, series, "quarterly", f"{year}Q{quarter}", part))
        employment = ("tri_434820_NSA.csv" if node == "rail_transport" else None)
        employment_parts = ([employment] if employment else NODE_NAICS[node].split(","))
        for year in range(2005, YEAR_END + 1):
            for month in range(1, 13):
                for part in employment_parts:
                    expected.append((node, "employment", "monthly", f"{year}-{month:02d}", part))
                if node == "funds_trusts":
                    expected.append((node, FUNDS_CES_MEASURE, "monthly",
                                     f"{year}-{month:02d}", FUNDS_CES_ID))
    audit = pd.DataFrame(expected, columns=["node_key", "series", "frequency", "period", "component"])
    actual = data[["node_key", "series", "frequency", "period", "component"]].drop_duplicates()
    audit = audit.merge(actual.assign(status="OBSERVED"),
                        on=["node_key", "series", "frequency", "period", "component"],
                        how="left", validate="one_to_one")
    audit["status"] = audit["status"].fillna("MISSING_SOURCE")
    return audit


def collect() -> dict:
    DATA.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    with requests.Session() as session:
        session.headers.update({"User-Agent": "Mozilla/5.0 PRISM research",
                                "Accept": "application/json, */*"})
        bea, bea_meta = collect_latest_bea(session)
        print(f"BEA latest release {bea_meta['release_date']}: {len(bea)} native observations", flush=True)
        rail, ces_meta = collect_latest_rail_ces(session)
        print(f"CES latest rail vintage {ces_meta['latest_vintage_month']}: {len(rail)} months", flush=True)
        funds, funds_meta = collect_latest_funds_ces(session)
        print(f"CES 523+525 current combined employment: {len(funds)} months", flush=True)
        qcew, qcew_meta = collect_latest_qcew(session)
        print(f"QCEW current API: {len(qcew)} native observations", flush=True)
    data = assign_nodes(bea, rail, qcew, funds)
    audit = coverage_audit(data)
    output = DATA / "latest_sources.csv.gz"
    data.to_csv(output, index=False, compression="gzip")
    audit.to_csv(RESULTS / "latest_source_coverage.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"node_key": node, "industry_name": industry, "naics_codes": NODE_NAICS[node]}
                  for node, industry, _ in INDUSTRIES]).to_csv(
        RESULTS / "node_registry.csv", index=False, encoding="utf-8-sig")
    missing = audit[audit["status"] != "OBSERVED"]
    report = {
        "status": "PASS" if missing.empty else "FAIL",
        "policy": "LATEST_OFFICIAL_VALUES_ONLY",
        "release_vintage_reconstruction": False,
        "target_computation": False,
        "industry_nodes": len(INDUSTRIES),
        "observation_years": [YEAR_START, YEAR_END],
        "latest_native_source_rows": len(data),
        "expected_required_native_source_cells": len(audit),
        "missing_required_native_source_cells": len(missing),
        "latest_file": str(output.relative_to(ROOT)),
        "bea": bea_meta, "ces": ces_meta, "funds_trusts_ces": funds_meta,
        "qcew": qcew_meta,
    }
    (RESULTS / "collection_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    if missing.empty:
        return report
    raise RuntimeError(f"Latest source coverage incomplete: {len(missing)} required cells")


if __name__ == "__main__":
    collect()
