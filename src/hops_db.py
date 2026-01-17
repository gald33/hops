import argparse
import csv
import json
import os


REGISTRY_PATH = os.path.join("runs", "_registry", "hops_registry.jsonl")


def load_records(path: str):
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def parse_filters(where: str):
    filters = []
    for chunk in where.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "~" in chunk:
            key, val = chunk.split("~", 1)
            filters.append((key.strip(), val.strip(), "contains"))
        elif "=" in chunk:
            key, val = chunk.split("=", 1)
            filters.append((key.strip(), val.strip(), "equals"))
    return filters


def apply_filters(records, filters):
    if not filters:
        return records
    out = []
    for rec in records:
        ok = True
        for key, val, mode in filters:
            raw = rec.get(key, "")
            raw_str = str(raw)
            if mode == "contains":
                if val not in raw_str:
                    ok = False
                    break
            else:
                if raw_str != val:
                    ok = False
                    break
        if ok:
            out.append(rec)
    return out


def write_csv(path: str, records):
    if not records:
        return
    fieldnames = sorted({k for rec in records for k in rec.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    list_cmd = sub.add_parser("list")
    list_cmd.add_argument("--last", type=int, default=20)
    query_cmd = sub.add_parser("query")
    query_cmd.add_argument("--where", type=str, default="")
    query_cmd.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    records = load_records(REGISTRY_PATH)
    if args.cmd == "list":
        last = records[-args.last :] if args.last else records
        for rec in last:
            print(json.dumps(rec, sort_keys=True))
        return
    filters = parse_filters(args.where)
    filtered = apply_filters(records, filters)
    if args.out:
        write_csv(args.out, filtered)
    else:
        for rec in filtered:
            print(json.dumps(rec, sort_keys=True))


if __name__ == "__main__":
    main()
