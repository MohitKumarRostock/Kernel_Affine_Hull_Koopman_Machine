"""Opaque source download and approved timestamp/operation-only projection.

This preparation never decodes, parses, prints or summarizes any count or
weather value. CSV data records are split as bytes; only Date, Hour and the
Functioning Day field are decoded. Quoted or malformed CSV rows are rejected
rather than passed to a whole-record text decoder.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import time
import urllib.request
import zipfile

ROOT=Path(__file__).resolve().parents[1]
URL="https://archive.ics.uci.edu/static/public/560/seoul%2Bbike%2Bsharing%2Bdemand.zip"
ARCHIVE=ROOT/"inputs/seoul_bike_uci560.zip"
ENTRY="SeoulBikeData.csv"

def sha_bytes(value):return hashlib.sha256(value).hexdigest()
def sha_file(path):return sha_bytes(Path(path).read_bytes())
def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n")

def download():
    if (ROOT/"protocol_seal38.json").exists():raise RuntimeError("Preparation forbidden after protocol freeze")
    if ARCHIVE.exists():raise RuntimeError("Opaque source already present; do not redownload")
    started=time.perf_counter()
    with urllib.request.urlopen(URL,timeout=45) as response:
        raw=response.read();resolved=response.geturl()
    ARCHIVE.parent.mkdir(parents=True,exist_ok=True);ARCHIVE.write_bytes(raw)
    receipt={"schema":"step38-opaque-public-source-download-v1","dataset":"Seoul Bike Sharing Demand",
        "uci_id":560,"source_url":URL,"resolved_url":resolved,"archive_path":str(ARCHIVE.relative_to(ROOT)),
        "archive_sha256":sha_bytes(raw),"archive_bytes":len(raw),
        "download_utc":datetime.now(timezone.utc).isoformat(),"download_elapsed_seconds":time.perf_counter()-started,
        "stage":"OPAQUE_ARCHIVE_ONLY","numerical_count_values_parsed":0,"weather_values_parsed":0,
        "timestamp_values_parsed":0,"source_license":"CC BY 4.0","dataset_doi":"10.24432/C5F62R",
        "metadata_url":"https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand",
        "citation":"Seoul Bike Sharing Demand [Dataset]. (2020). UCI Machine Learning Repository. https://doi.org/10.24432/C5F62R.",
        "absolute_historical_never_seen_claim":False,"scientific_model_fits":0}
    write_json(ROOT/"inputs/public_source38.json",receipt)
    return receipt

def calendar_projection(raw_csv):
    records=raw_csv.splitlines(keepends=True)
    if not records:raise RuntimeError("Empty CSV")
    header_raw=records[0].rstrip(b"\r\n").removeprefix(b"\xef\xbb\xbf")
    if b'"' in header_raw:raise RuntimeError("Quoted CSV schema is outside approved binary projection")
    header_fields=header_raw.split(b",")
    # Whole-header decoding exposes field names only, never data-row values.
    headers=[value.decode("latin1") for value in header_fields]
    if (len(headers)!=14 or headers[0]!="Date" or headers[1]!="Rented Bike Count" or headers[2]!="Hour"
        or headers[13] not in ("Functioning Day","Functional Day")):
        raise RuntimeError("Official schema differs from approved field projection")
    rows=[];offset=len(records[0]);seen=set();last=None
    for line_number,raw_record in enumerate(records[1:],2):
        body=raw_record.rstrip(b"\r\n")
        if not body:raise RuntimeError("Unexpected empty record")
        if b'"' in body:raise RuntimeError("Quoted record requires a separately approved binary parser")
        cells=body.split(b",")
        if len(cells)!=14:raise RuntimeError("Malformed field count")
        date_text=cells[0].decode("ascii");hour_text=cells[2].decode("ascii");function_text=cells[13].decode("ascii")
        if not re.fullmatch(r"\d{2}/\d{2}/\d{4}",date_text) or not re.fullmatch(r"\d{1,2}",hour_text):
            raise RuntimeError("Date/hour format differs from declared day-first projection")
        hour=int(hour_text)
        if not 0<=hour<=23:raise RuntimeError("Invalid hour")
        stamp=datetime.strptime(date_text,"%d/%m/%Y").replace(hour=hour)
        iso=stamp.isoformat(timespec="seconds")
        if iso in seen or (last is not None and stamp<=last):raise RuntimeError("Duplicate or unordered calendar row")
        seen.add(iso);last=stamp
        if function_text not in ("Yes","No","Fun","NoFunc"):
            raise RuntimeError("Unknown operation flag")
        rows.append({"timestamp":iso,"functioning_day":function_text,"csv_line":line_number,
                     "byte_start":offset,"byte_stop":offset+len(raw_record)})
        offset+=len(raw_record)
    first=datetime.fromisoformat(rows[0]["timestamp"]);last=datetime.fromisoformat(rows[-1]["timestamp"])
    expected=[];current=first
    while current<=last:expected.append(current.isoformat(timespec="seconds"));current+=timedelta(hours=1)
    missing=[stamp for stamp in expected if stamp not in seen]
    return {"schema":"step38-calendar-only-byte-projection-v1","headers":headers,
        "header_encoding":"latin1","parsed_data_columns":[0,2,13],"parsed_data_field_names":[headers[i] for i in (0,2,13)],
        "count_field_index":1,"data_record_decoding":"ASCII for allowed fields only; other fields retained as opaque bytes",
        "date_format":"%d/%m/%Y","source_timezone_convention":"local naive source date/hour; no UTC reinterpretation",
        "csv_bytes":len(raw_csv),"csv_sha256":sha_bytes(raw_csv),"rows":rows,
        "timestamp_rows":len(rows),"first_timestamp":rows[0]["timestamp"],"last_timestamp":rows[-1]["timestamp"],
        "expected_calendar_hours":len(expected),"missing_calendar_timestamps":missing,"duplicate_timestamps":0,
        "operation_flag_counts":dict(sorted(Counter(row["functioning_day"] for row in rows).items())),
        "count_values_decoded_or_parsed":0,"weather_values_decoded_or_parsed":0,
        "closure_rows_retained":True,"operation_flags_used_as_predictors":False}

def project():
    if (ROOT/"protocol_seal38.json").exists():raise RuntimeError("Preparation forbidden after protocol freeze")
    source=json.loads((ROOT/"inputs/public_source38.json").read_text())
    if sha_file(ARCHIVE)!=source["archive_sha256"]:raise RuntimeError("Opaque archive changed")
    with zipfile.ZipFile(io.BytesIO(ARCHIVE.read_bytes())) as archive:
        names=archive.namelist()
        if names!=[ENTRY]:raise RuntimeError("Unexpected archive member roster")
        raw=archive.read(ENTRY)
    result=calendar_projection(raw)
    result.update({"archive_sha256":source["archive_sha256"],"archive_member":ENTRY,
        "approved_parent_scope":"Date,Hour,FunctioningDay binary projection only; count/weather fields never decoded",
        "projection_utc":datetime.now(timezone.utc).isoformat(),"projection_source_sha256":sha_file(Path(__file__))})
    write_json(ROOT/"inputs/calendar_projection38.json",result)
    metadata={key:value for key,value in result.items() if key!="rows"}
    metadata["projection_sha256"]=sha_file(ROOT/"inputs/calendar_projection38.json")
    metadata["metadata_reported_instances"]=8760
    metadata["metadata_count_matches_actual_timestamp_rows"]=len(result["rows"])==8760
    metadata["official_date_description_differs_from_file"]=True
    metadata["official_operation_tokens_differ_from_file"]=True
    write_json(ROOT/"inputs/calendar_metadata38.json",metadata)
    return metadata

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("mode",choices=("download","project"));args=parser.parse_args()
    result=download() if args.mode=="download" else project()
    print(json.dumps(result,sort_keys=True,allow_nan=False))
