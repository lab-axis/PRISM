#!/usr/bin/env python3
"""Collect the current/revised official values of PRISM predictor sources.

Same latest-value policy as Stage 01: no archival-vintage reconstruction,
publication calendars, as-of substitution or rolling feature expansion.
Each native statistical series is ONE candidate channel; industry scope and
original periods are preserved, with explicit masks for unpublished series.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "Data/02_initial_feature_pool"
RAW_DIR = DATA_DIR / "raw_latest"
METADATA_DIR = RAW_DIR / "metadata"
METADATA_MANIFEST = METADATA_DIR / "source_manifest.json"
PROC_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT / "Results/02_initial_feature_pool"
TARGET_LATEST = ROOT / "Data/01_target_data_collection/latest_sources.csv.gz"
TARGET_SUMMARY = ROOT / "Results/01_target_data_collection/collection_summary.json"
NODE_REGISTRY = ROOT / "Results/01_target_data_collection/node_registry.csv"
QUARTERS = pd.period_range("2005Q1", "2025Q4", freq="Q").astype(str).tolist()
BLS_API = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
BLS_METADATA = "https://download.bls.gov/pub/time.series"
NYFED_EFFR = "https://markets.newyorkfed.org/api/rates/unsecured/effr/search.json"
FRED_ALIASES = {
    "CUUR0000SA0": "CPIAUCNS", "CUUR0000SA0L1E": "CPILFENS",
    "CUUR0000SA0E": "CPIENGNS", "CUUR0000SAF1": "CPIUFDNS",
    "LNU04000000": "UNRATENSA", "CEU0000000001": "PAYNSA",
}

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

def node_stats(df: pd.DataFrame) -> tuple[float, float, float]:
    per_node = df.notna().mean(axis=0)
    observed = per_node[per_node > 0]
    node_fraction = float((per_node > 0).mean())
    median_time = float(observed.median()) if not observed.empty else 0.0
    mean_all = float(df.notna().mean().mean())
    return node_fraction, median_time, mean_all

def register_node(
    blocks: dict[str, pd.DataFrame], source: str, group: str, registry: list[dict]
) -> dict[str, pd.DataFrame]:
    """One observed source measure -> one feature. No automatic expansion."""
    for base_name, df in blocks.items():
        nf, mt, ma = node_stats(df)
        registry.append({
            "feature_id": base_name, "scope": "NODE",
            "source": source, "group": group,
            "node_fraction": nf, "median_time_coverage": mt, "overall_coverage": ma,
            "status": "PASS",
        })
    return dict(blocks)

def register_global(
    blocks: dict[str, pd.Series], source: str, group: str, registry: list[dict]
) -> dict[str, pd.Series]:
    """One original macro/market series -> one feature, with no derivatives."""
    for base_name, s in blocks.items():
        cov = float(s.notna().mean())
        registry.append({
            "feature_id": base_name, "scope": "GLOBAL",
            "source": source, "group": group,
            "node_fraction": 1.0, "median_time_coverage": cov, "overall_coverage": cov,
            "status": "PASS",
        })
    return dict(blocks)

def validate_native_feature_inventory(
    node_features: dict[str, pd.DataFrame],
    global_features: dict[str, pd.Series],
    source_mapping: list[dict],
    registry: list[dict],
) -> None:
    """Prevent accidental restoration of the old feature-multiplication code.

    Independently published CES measures (employment, hours, earnings, etc.)
    count as distinct native features. Transformations (deltas, log-deltas,
    returns, volatility) of one source must NEVER become extra features.
    """
    node_ids, global_ids = set(node_features), set(global_features)
    feature_ids = node_ids | global_ids
    if node_ids & global_ids:
        raise RuntimeError("Industry and nationwide feature names overlap")
    if (len(node_ids), len(global_ids)) != (24, 91):
        raise RuntimeError(
            f"Unexpected Stage 02 inventory {len(node_ids)}+{len(global_ids)}; "
            "only 24+91 original source measures are permitted"
        )
    derivative_names = [
        name for name in feature_ids
        if name.endswith(("__d1", "__d4", "__logd1", "__logd4", "_qret", "_vol"))
        or (name.startswith("market_") and name.endswith("_level"))
    ]
    if derivative_names:
        raise RuntimeError(f"Synthetic feature expansion detected: {derivative_names}")
    registered = [item["feature_id"] for item in registry]
    if len(registered) != len(feature_ids) or set(registered) != feature_ids:
        raise RuntimeError("Native registry must match feature IDs one-to-one")
    original_to_feature: dict[tuple[str, str], str] = {}
    for item in source_mapping:
        feature_id = item["feature_id"]
        if feature_id not in node_ids:
            raise RuntimeError(f"Industry source mapping has unknown feature: {feature_id}")
        source_key = (item["source"], item["series_id"])
        original = original_to_feature.setdefault(source_key, feature_id)
        if original != feature_id:
            raise RuntimeError(
                f"One native source series produced multiple features: {source_key}"
            )


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

def directories() -> None:
    for path in (RAW_DIR, PROC_DIR, RESULTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def context() -> tuple[list[dict], list[str]]:
    if not TARGET_LATEST.exists() or not NODE_REGISTRY.exists():
        raise FileNotFoundError("Run the latest-value Stage 01 collector first")
    summary = json.loads(TARGET_SUMMARY.read_text(encoding="utf-8"))
    if summary.get("status") != "PASS" or summary.get("policy") != "LATEST_OFFICIAL_VALUES_ONLY":
        raise RuntimeError("Stage 01 latest-value source collection has not passed")
    df = pd.read_csv(NODE_REGISTRY, dtype=str).fillna("")
    if len(df) != 56 or df["node_key"].duplicated().any():
        raise RuntimeError("The fixed 56-industry node registry is invalid")
    return [
        {"key": row["node_key"], "name": row["industry_name"],
         "naics_codes": row["naics_codes"]}
        for row in df.to_dict("records")
    ], QUARTERS


def bls_metadata(family: str, file: str) -> pd.DataFrame:
    """Official BLS reference catalog, independent of old Git repository files.

    Each preserved catalog has a SHA-256 entry in its source manifest.
    A missing catalog is downloaded from BLS; a 403 without an archived
    catalog is a hard error, never an excuse to query an older Git commit.
    """
    catalog = METADATA_DIR / file
    if not catalog.is_file():
        url = f"{BLS_METADATA}/{family}/{file}"
        response = requests.get(url, timeout=(8, 30))
        response.raise_for_status()
        METADATA_DIR.mkdir(parents=True, exist_ok=True)
        catalog.write_bytes(response.content)
        metadata_manifest = (json.loads(METADATA_MANIFEST.read_text(encoding="utf-8"))
                             if METADATA_MANIFEST.exists()
                             else {"source_type": "official_BLS_data_snapshot",
                                   "runtime_git_dependency": False, "files": {}})
        metadata_manifest["files"][catalog.relative_to(ROOT).as_posix()] = {
            "source_url": url, "sha256": hashlib.sha256(response.content).hexdigest(),
            "bytes": len(response.content), "origin": "live_BLS_catalog_download",
        }
        METADATA_MANIFEST.write_text(
            json.dumps(metadata_manifest, indent=2) + "\n", encoding="utf-8")
    data = catalog.read_bytes()
    if not METADATA_MANIFEST.is_file():
        raise RuntimeError("Official BLS source metadata manifest is missing")
    manifest = json.loads(METADATA_MANIFEST.read_text(encoding="utf-8"))
    entry = manifest.get("files", {}).get(catalog.relative_to(ROOT).as_posix())
    if entry is None or entry.get("sha256") != hashlib.sha256(data).hexdigest():
        raise RuntimeError(f"Official BLS catalog source checksum mismatch: {catalog}")
    metadata = pd.read_csv(io.BytesIO(data), sep="\t", dtype=str).fillna("")
    metadata.columns = [column.strip() for column in metadata]
    for column in metadata:
        metadata[column] = metadata[column].astype(str).str.strip()
    return metadata


def current_bls(session: requests.Session, family: str, source_ids: set[str],
                *, refresh: bool = False) -> tuple[pd.DataFrame, dict]:
    """Native BLS time series, with no as-of snapshots or synthetic lags."""
    source_ids = set(source_ids)
    if not source_ids:
        return pd.DataFrame(columns=["series_id", "year", "period", "value"]), {
            "source": family, "series": 0, "origin": "empty_official_series_catalog"}
    path = RAW_DIR / f"{family}.csv"
    origin = "previous_latest_run"
    if path.exists():
        df = pd.read_csv(path, dtype={"series_id": str, "period": str})
        origin = "portable_official_source_cache"
    else:
        df = pd.DataFrame(columns=["series_id", "year", "period", "value"])
        origin = "BLS_current_API"
    if refresh or df.empty:
        rows = []
        # Re-fetch only on explicit refresh or when there is no portable cache.
        # Incomplete API batches are not silently presented as a full refresh.
        exhausted = False
        for start in range(2003, 2026, 10):
            end = min(2025, start + 9)
            for i in range(0, len(source_ids), 20):
                batch = sorted(source_ids)[i:i + 20]
                try:
                    response = session.post(
                        BLS_API, json={"seriesid": batch, "startyear": str(start),
                                       "endyear": str(end)}, timeout=60,
                    )
                    response.raise_for_status()
                    payload = response.json()
                    if payload.get("status") != "REQUEST_SUCCEEDED":
                        raise RuntimeError(
                            f"BLS latest data unavailable: {payload.get('message')}")
                except (requests.RequestException, RuntimeError) as exc:
                    if df.empty:
                        raise RuntimeError(
                            f"No portable {family} BLS source cache and live API unavailable"
                        ) from exc
                    exhausted = True
                    break
                for source in payload.get("Results", {}).get("series", []):
                    for item in source.get("data", []):
                        if not re.fullmatch(r"M(?:0[1-9]|1[0-2])",
                                            item.get("period", "")):
                            continue
                        try:
                            val = float(str(item["value"]).replace(",", ""))
                        except (TypeError, ValueError):
                            continue
                        if np.isfinite(val):
                            rows.append({"series_id": source["seriesID"],
                                         "year": int(item["year"]),
                                         "period": item["period"], "value": val})
            if exhausted:
                break
        if not exhausted and rows:
            fresh = pd.DataFrame(rows)
            df = pd.concat([df, fresh], ignore_index=True)
            df = df.drop_duplicates(["series_id", "year", "period"], keep="last")
            origin = "complete_BLS_current_API_update"
        elif exhausted:
            origin = "portable_official_source_cache_API_unavailable"
        # If not enough BLS data were available, preserve the old validated
        # cache. Never replace it with an incomplete partial API response.
        if not df.empty:
            df.to_csv(path, index=False)
    df = df[df["series_id"].isin(source_ids)].copy()
    # Some stored BLS flat-file extracts contain only a subset of the national
    # CPI indicators. Fill those from their currently published identical
    # unadjusted FRED/BLS source series, rather than leave invented gaps.
    if family == "cpi":
        added = []
        for sid in sorted(source_ids):
            fred = FRED_ALIASES.get(sid, sid)
            response = None
            last_error = None
            for attempt in range(3):
                try:
                    # FRED's CSV endpoint rejects certain browser-like user
                    # agents that BLS needs. Use the default requests identity
                    # for this public export, not our BLS session headers.
                    response = requests.get(
                        "https://fred.stlouisfed.org/graph/fredgraph.csv",
                        params={"id": fred}, timeout=(6, 15),
                    )
                    response.raise_for_status()
                    break
                except requests.RequestException as exc:
                    last_error = exc
            else:
                raise RuntimeError(f"Latest CPI source unavailable after retries: {fred}") from last_error
            table = pd.read_csv(io.BytesIO(response.content))
            date_col = ("observation_date" if "observation_date" in table
                        else "DATE" if "DATE" in table else None)
            if date_col is None or fred not in table:
                raise RuntimeError(f"Missing official current CPI source: {fred}")
            dates = pd.to_datetime(table[date_col], errors="raise")
            if not dates.dt.day.eq(1).all():
                raise RuntimeError(f"Unexpected CPI source frequency: {fred}")
            for period, value in zip(dates.dt.to_period("M"), table[fred]):
                try:
                    numeric = float(value)
                except (ValueError, TypeError):
                    continue
                if 2003 <= period.year <= 2025 and np.isfinite(numeric):
                    added.append({"series_id": sid, "year": period.year,
                                  "period": f"M{period.month:02d}", "value": numeric})
        if added:
            df = pd.concat([df, pd.DataFrame(added)], ignore_index=True)
            df = df.drop_duplicates(["series_id", "year", "period"], keep="last")
            df.to_csv(path, index=False)
            origin += "+current_FRED_CPI"
    df["year"] = pd.to_numeric(df["year"], errors="raise").astype(int)
    df["value"] = pd.to_numeric(df["value"], errors="raise")
    df = df[df["year"].between(2003, 2025)]
    if df.duplicated(["series_id", "year", "period"]).any():
        raise RuntimeError(f"Duplicate BLS native observation: {family}")
    if not np.isfinite(df["value"]).all():
        raise RuntimeError(f"Nonfinite BLS source value: {family}")
    return df, {"source": family, "series_requested": len(source_ids),
                "series_obtained": int(df["series_id"].nunique()),
                "native_observations": len(df), "origin": origin}


def quarterly_bls(monthly: pd.DataFrame, quarters: list[str]) -> dict[str, pd.Series]:
    """One quarterly mean per original monthly series, requiring three months."""
    if monthly.empty:
        return {}
    d = monthly[monthly["period"].str.fullmatch(r"M(?:0[1-9]|1[0-2])")].copy()
    d["quarter"] = (d["year"].astype(int).astype(str) + "Q"
                    + (((d["period"].str[1:].astype(int) - 1) // 3) + 1).astype(str))
    q = d.groupby(["series_id", "quarter"]).value.agg(["mean", "count"]).reset_index()
    q.loc[q["count"] != 3, "mean"] = np.nan
    return {sid: g.set_index("quarter")["mean"].reindex(quarters).astype(float)
            for sid, g in q.groupby("series_id")}


def latest_qcew_employment(nodes: list[dict], quarters: list[str]) -> dict[str, pd.DataFrame]:
    """Read *latest* Stage 01 monthly employment, not historical as-of views."""
    source = pd.read_csv(TARGET_LATEST, low_memory=False,
                         dtype={"period": str, "source_industry": str, "node_key": str})
    source = source[source["series"] == "employment"].copy()
    source["value"] = pd.to_numeric(source["value"], errors="raise")
    source["value_thousands"] = np.where(
        source["units"] == "persons", source["value"] / 1000.0, source["value"])
    matrix = pd.DataFrame(np.nan, index=quarters, columns=[n["key"] for n in nodes])
    for node in nodes:
        current = source[source["node_key"] == node["key"]]
        expected = (["tri_434820_NSA.csv"] if node["key"] == "rail_transport"
                    else node["naics_codes"].split(","))
        raw = current.pivot(index="period", columns="source_industry", values="value_thousands")
        native = raw.reindex(columns=expected)
        # An industry composed of multiple NAICS parts needs EVERY part.
        monthly = native.sum(axis=1, min_count=len(expected))
        counts = native.notna().sum(axis=1)
        monthly = monthly.where(counts == len(expected))
        ref = pd.PeriodIndex(monthly.index, freq="M").asfreq("Q").astype(str)
        grp = monthly.groupby(ref)
        values = grp.mean().where(grp.count() == 3)
        matrix[node["key"]] = values.reindex(quarters).to_numpy()
    return {"qcew_employment": matrix}


def latest_ces(session: requests.Session, nodes: list[dict], quarters: list[str],
               refresh: bool, evidence: list) -> tuple[dict, list[dict]]:
    catalog = bls_metadata("ce", "ce.series")
    industry = bls_metadata("ce", "ce.industry")
    permitted = set(catalog["series_id"])
    industry_code: dict[str, str] = {}
    for node in nodes:
        target = node_naics_set(node)
        found = industry[industry["naics_code"].map(
            lambda spec: expand_compact_naics(spec) == target)]
        if not found.empty:
            choices = found.copy()
            choices["level"] = pd.to_numeric(choices["display_level"], errors="coerce")
            industry_code[node["key"]] = str(choices.sort_values("level").iloc[0]["industry_code"])
    mapping = {dtype: {} for dtype in CES_TYPES}
    for dtype in CES_TYPES:
        for key, code in industry_code.items():
            sid = f"CEU{code}{dtype}"
            if sid in permitted:
                mapping[dtype][key] = sid
    wanted = set(GLOBAL_CES)
    for sub in mapping.values():
        wanted.update(sub.values())
    monthly, provenance = current_bls(session, "ces_industry",
                                      wanted, refresh=refresh)
    evidence.append(provenance)
    quarterly = quarterly_bls(monthly, quarters)
    features, links = {}, []
    keys = [node["key"] for node in nodes]
    for dtype, label in CES_TYPES.items():
        name = "ces_exact_" + label
        panel = pd.DataFrame(np.nan, index=quarters, columns=keys)
        for node_key, sid in mapping[dtype].items():
            links.append({"source": "BLS CES", "feature_id": name,
                          "node_key": node_key, "series_id": sid})
            if sid in quarterly:
                panel[node_key] = quarterly[sid].to_numpy()
        features[name] = panel
    return features, links


def latest_ppi(session: requests.Session, nodes: list[dict], quarters: list[str],
               refresh: bool, evidence: list) -> tuple[dict, list[dict]]:
    meta = bls_metadata("pc", "pc.series")
    top = meta[(meta["industry_code"] == meta["product_code"])
               & meta["series_id"].str.startswith("PCU")].copy()
    mapping = {}
    for node in nodes:
        target = node_naics_set(node)
        candidates = top[top["industry_code"].map(
            lambda spec: expand_compact_naics(str(spec).rstrip("-")) == target)]
        if candidates.empty:
            continue
        candidates = candidates.copy()
        candidates["first_year"] = pd.to_numeric(
            candidates["begin_year"], errors="coerce").fillna(9999)
        mapping[node["key"]] = str(candidates.sort_values("first_year").iloc[0]["series_id"])
    monthly, provenance = current_bls(session, "ppi_industry",
                                      set(mapping.values()),
                                      refresh=refresh)
    evidence.append(provenance)
    quarterly = quarterly_bls(monthly, quarters)
    panel = pd.DataFrame(np.nan, index=quarters, columns=[n["key"] for n in nodes])
    links = []
    for key, sid in mapping.items():
        links.append({"source": "BLS PPI", "feature_id": "ppi_industry_net_output",
                      "node_key": key, "series_id": sid})
        if sid in quarterly:
            panel[key] = quarterly[sid].to_numpy()
    return {"ppi_industry_net_output": panel}, links


def latest_jolts(session: requests.Session, nodes: list[dict], quarters: list[str],
                 refresh: bool, evidence: list) -> tuple[dict, dict, list[dict]]:
    meta = bls_metadata("jt", "jt.series")
    meta = meta[(meta["seasonal"] == "U") & (meta["state_code"] == "00")
                & (meta["area_code"] == "00000") & (meta["sizeclass_code"] == "00")]
    industries = {jolts_industry(n["key"]) for n in nodes}
    eligible = meta[
        meta["industry_code"].isin(industries | {"000000"})
        & meta["dataelement_code"].isin(set(JOLTS_ELEMENTS) | {"UO"})
        & meta["ratelevel_code"].isin({"L", "R"})
    ].copy()
    monthly, provenance = current_bls(
        session, "jolts", set(eligible["series_id"]),
        refresh=refresh)
    evidence.append(provenance)
    quarterly = quarterly_bls(monthly, quarters)
    node_keys = [n["key"] for n in nodes]
    node_out, global_out, mapping = {}, {}, []
    for element, label in JOLTS_ELEMENTS.items():
        for code, suffix in (("R", "rate"), ("L", "level")):
            base = f"jolts_{label}_{suffix}"
            panel = pd.DataFrame(np.nan, index=quarters, columns=node_keys)
            for node in nodes:
                hits = eligible[
                    (eligible["industry_code"] == jolts_industry(node["key"]))
                    & (eligible["dataelement_code"] == element)
                    & (eligible["ratelevel_code"] == code)
                ]
                if hits.empty:
                    continue
                sid = str(hits.iloc[0]["series_id"])
                mapping.append({"source": "BLS JOLTS", "feature_id": base,
                                "node_key": node["key"], "series_id": sid})
                if sid in quarterly:
                    panel[node["key"]] = quarterly[sid].to_numpy()
            node_out[base] = panel
    total = eligible[eligible["industry_code"] == "000000"]
    for element, label in {**JOLTS_ELEMENTS, "UO": "unemployed_per_opening"}.items():
        for code, suffix in (("R", "rate"), ("L", "level")):
            hits = total[(total["dataelement_code"] == element)
                         & (total["ratelevel_code"] == code)]
            if not hits.empty:
                sid = str(hits.iloc[0]["series_id"])
                if sid in quarterly:
                    global_out[f"bls_jolts_{label}_{suffix}"] = quarterly[sid]
    return node_out, global_out, mapping


def latest_national_bls(session: requests.Session, quarters: list[str],
                        refresh: bool, evidence: list) -> dict[str, pd.Series]:
    configs = [
        ("cpi", GLOBAL_CPI),
        ("cps", GLOBAL_CPS),
        ("national_ces", GLOBAL_CES),
        ("ppi_commodities", {"WPU00000000": GLOBAL_PPI["WPU00000000"]}),
        ("ppi_final_demand", {k: v for k, v in GLOBAL_PPI.items()
                              if k != "WPU00000000"}),
    ]
    out = {}
    for group, series in configs:
        source, provenance = current_bls(
            session, group, set(series), refresh=refresh)
        evidence.append(provenance)
        quarterly = quarterly_bls(source, quarters)
        for sid, label in series.items():
            out["bls_" + label] = quarterly.get(
                sid, pd.Series(np.nan, index=quarters, dtype=float))
    return out


def latest_nyfed(session: requests.Session, quarters: list[str]) -> dict[str, pd.Series]:
    response = session.get(NYFED_EFFR, params={
        "startDate": "2005-01-01", "endDate": "2025-12-31", "type": "rate"
    }, timeout=70)
    response.raise_for_status()
    table = pd.DataFrame(response.json().get("refRates", []))
    if table.empty:
        raise RuntimeError("New York Fed latest EFFR API returned no data")
    table["quarter"] = pd.to_datetime(table["effectiveDate"]).dt.to_period("Q").astype(str)
    out = {}
    for column, label in [
        ("percentRate", "nyfed_effr"),
        ("targetRateFrom", "nyfed_target_lower"),
        ("targetRateTo", "nyfed_target_upper"),
    ]:
        if column in table:
            table[column] = pd.to_numeric(table[column], errors="coerce")
            out[label] = table.groupby("quarter")[column].mean().reindex(quarters)
    return out


def latest_market(quarters: list[str], refresh: bool) -> tuple[dict[str, pd.Series], list[dict]]:
    """Native unadjusted closing price, one channel for each original ticker."""
    cache = RAW_DIR / "market_closes.csv"
    origin = "portable_market_close_source_cache"
    if cache.exists() and not refresh:
        prices = pd.read_csv(cache, index_col=0, parse_dates=True)
    else:
        # A clean checkout needs either its explicit portable source CSV or
        # a successful public download; never read a deleted Git file.
        try:
            import yfinance as yf
            history = yf.download(
                list(MARKET_TICKERS), start="2005-01-01", end="2026-01-01",
                progress=False, auto_adjust=False, threads=False,
                group_by="column", timeout=20,
            )
            if history.empty or "Close" not in history:
                raise RuntimeError("Market download lacks the requested Close prices")
            prices = history["Close"]
            if isinstance(prices, pd.Series):
                prices = prices.to_frame()
            if not set(MARKET_TICKERS).issubset(prices.columns):
                raise RuntimeError("Current market download lacks required tickers")
        except (ImportError, RuntimeError, ValueError, KeyError,
                requests.RequestException) as exc:
            raise RuntimeError(
                "No usable portable market close source; live market download failed"
            ) from exc
        origin = "public_market_close_download"
        prices.to_csv(cache)
    # Complete the ordinary-price baseline through December 2025. The
    # archived project CSV may end earlier, but a latest-data experiment
    # must not leave published recent quarters empty.
    end_of_history = pd.to_datetime(prices.index).max()
    if refresh or end_of_history < pd.Timestamp("2025-10-01"):
        import yfinance as yf
        tail = yf.download(
            list(MARKET_TICKERS),
            start="2025-01-01", end="2026-01-01",
            auto_adjust=False, progress=False, threads=False,
            group_by="column", timeout=15,
        )
        if tail.empty or "Close" not in tail:
            raise RuntimeError("Latest Yahoo close-price history is unavailable")
        current = tail["Close"]
        if isinstance(current, pd.Series):
            current = current.to_frame()
        prices = pd.concat([prices, current], axis=0).sort_index()
        prices = prices.loc[~prices.index.duplicated(keep="last")]
        prices.to_csv(cache)
        origin += "+latest_2025_market_prices"
    if not set(MARKET_TICKERS).issubset(prices):
        raise RuntimeError("Latest market source has missing ticker columns")
    prices.index = pd.to_datetime(prices.index)
    qindex = pd.PeriodIndex(quarters, freq="Q")
    out, audit = {}, []
    for ticker, label in MARKET_TICKERS.items():
        series = pd.to_numeric(prices[ticker], errors="coerce").dropna()
        if series.empty:
            raise RuntimeError(f"No latest source observations for ticker {ticker}")
        close = series.groupby(series.index.to_period("Q")).last().reindex(qindex)
        close.index = quarters
        out["market_" + label] = close
        audit.append({"ticker": ticker, "label": label, "source": origin,
                      "first_trading_date": str(series.index.min().date()),
                      "last_trading_date": str(series.index.max().date())})
    return out, audit


def coverage_audit(node: dict[str, pd.DataFrame], globals_: dict[str, pd.Series],
                   mapping: list[dict], quarters: list[str], nodes: list[dict],
                   market_info: list[dict]) -> pd.DataFrame:
    links = {(item["feature_id"], item["node_key"]) for item in mapping}
    audit = []
    for name, table in node.items():
        if list(table.index) != quarters or list(table.columns) != [n["key"] for n in nodes]:
            raise RuntimeError(f"Misaligned latest industry source: {name}")
        for key in table:
            first = next((q for q, v in table[key].items() if pd.notna(v)), None)
            exact = name.startswith(("ces_exact_", "ppi_industry_"))
            for quarter, val in table[key].items():
                status = ("OBSERVED" if pd.notna(val)
                          else "NOT_PUBLISHED_FOR_INDUSTRY" if exact and (name, key) not in links
                          else "NOT_YET_PUBLISHED" if first and quarter < first
                          else "COLLECTION_GAP")
                audit.append({"quarter": quarter, "node_key": key, "measure": name,
                              "status": status})
    launches = {item["label"]: item["first_trading_date"] for item in market_info}
    # October 2025 was not collected for these CPI/CPS sources during the
    # federal shutdown. An actual 2025Q4 3-month mean cannot be constructed;
    # missing official data must not be fabricated or called a download error.
    missing_october_2025 = {
        "bls_" + label for label in [*GLOBAL_CPI.values(), *GLOBAL_CPS.values()]
    }
    for name, source in globals_.items():
        if list(source.index) != quarters:
            raise RuntimeError(f"Misaligned latest national source: {name}")
        first = next((q for q, val in source.items() if pd.notna(val)), None)
        for quarter, val in source.items():
            status = ("OBSERVED" if pd.notna(val)
                      else "OFFICIALLY_NOT_COLLECTED"
                      if quarter == "2025Q4" and name in missing_october_2025
                      else "NOT_YET_PUBLISHED" if first and quarter < first
                      else "COLLECTION_GAP")
            evidence = ("https://www.bls.gov/bls/2025-lapse-revised-release-dates.htm"
                        if status == "OFFICIALLY_NOT_COLLECTED" else "")
            audit.append({"quarter": quarter, "node_key": "ALL",
                          "measure": name, "status": status, "evidence": evidence})
    return pd.DataFrame(audit)


def collect(*, refresh: bool = False) -> dict:
    directories()
    nodes, quarters = context()
    (RESULTS_DIR / "feature_pool_summary.json").write_text(
        json.dumps({"status": "RUNNING"}), encoding="utf-8")
    evidence: list[dict] = []
    with requests.Session() as session:
        session.headers["User-Agent"] = "Mozilla/5.0 PRISM academic research"
        print("[1/6] Latest QCEW/CES industrial employment", flush=True)
        qcew = latest_qcew_employment(nodes, quarters)
        ces, ces_mapping = latest_ces(session, nodes, quarters, refresh, evidence)
        print("[2/6] Latest industry PPI", flush=True)
        ppi, ppi_mapping = latest_ppi(session, nodes, quarters, refresh, evidence)
        print("[3/6] Latest JOLTS original source series", flush=True)
        jolts, jolts_global, jolts_mapping = latest_jolts(
            session, nodes, quarters, refresh, evidence)
        print("[4/6] Latest national BLS/New York Fed series", flush=True)
        national = latest_national_bls(session, quarters, refresh, evidence)
        nyfed = latest_nyfed(session, quarters)
    print("[5/6] Latest market original close", flush=True)
    market, market_audit = latest_market(quarters, refresh)
    node_blocks = {**qcew, **ces, **ppi, **jolts}
    global_blocks = {**national, **jolts_global, **nyfed, **market}
    mapping = ces_mapping + ppi_mapping + jolts_mapping
    registry = []
    node_features = register_node(node_blocks, "BLS industry", "NODE", registry)
    global_features = register_global(global_blocks, "BLS/NYFed/Market", "GLOBAL", registry)
    expected = len(node_blocks) + len(global_blocks)
    if len(node_features) + len(global_features) != expected:
        raise RuntimeError("Feature duplication or missing original feature")
    validate_native_feature_inventory(node_features, global_features, mapping, registry)
    print("[6/6] Validate all source channels", flush=True)
    audit = coverage_audit(node_features, global_features, mapping,
                           quarters, nodes, market_audit)
    write_node_long(PROC_DIR / "node_features_quarterly_long.csv", quarters, nodes,
                    node_features)
    write_node_masks(PROC_DIR / "node_feature_masks_quarterly_long.csv",
                     quarters, nodes, node_features)
    pd.DataFrame({"quarter": quarters, **{key: s.to_numpy()
                                         for key, s in global_features.items()}}).to_csv(
        PROC_DIR / "global_features_quarterly.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(registry).to_csv(
        RESULTS_DIR / "initial_feature_registry.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(mapping).to_csv(
        RESULTS_DIR / "node_source_mapping.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(market_audit).to_csv(
        RESULTS_DIR / "market_source_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(evidence).to_csv(
        RESULTS_DIR / "latest_source_provenance.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(RESULTS_DIR / "feature_coverage_audit.csv", index=False, encoding="utf-8-sig")
    problems = int((audit.status == "COLLECTION_GAP").sum())
    summary = {
        "status": "PASS" if problems == 0 else "FAIL",
        "policy": "LATEST_OFFICIAL_VALUES_ONLY",
        "publication_vintage_reconstruction": False,
        "automatic_feature_expansion": False,
        "derived_feature_count": 0,
        "period": [quarters[0], quarters[-1]],
        "quarters": len(quarters), "nodes": len(nodes),
        "node_features": len(node_features),
        "global_features": len(global_features),
        "total_original_features": expected,
        "coverage_counts": audit.status.value_counts().to_dict(),
        "unresolved_published_source_gaps": problems,
    }
    (RESULTS_DIR / "feature_pool_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if problems:
        raise RuntimeError(f"{problems} latest published-source gaps remain")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true",
                        help="refresh the latest values from source APIs where available")
    args = parser.parse_args()
    try:
        collect(refresh=args.refresh)
    except Exception as exc:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / "feature_pool_summary.json").write_text(
            json.dumps({"status": "FAIL", "error": str(exc)}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        raise
