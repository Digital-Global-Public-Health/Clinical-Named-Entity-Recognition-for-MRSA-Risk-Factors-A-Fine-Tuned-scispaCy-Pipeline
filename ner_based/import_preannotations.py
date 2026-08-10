#!/usr/bin/env python3
"""
Import WebAnno TSV3 pre-annotations into an INCEpTION project via the AERO
remote API. Run ON A MINERVA COMPUTE NODE against the local server so no PHI
leaves the enclave.
"""
import argparse, sys, glob, os
import requests
from requests.auth import HTTPBasicAuth

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--user", required=True)
    ap.add_argument("--password", required=True)
    ap.add_argument("--project-id", required=True)
    ap.add_argument("--tsv-dir", required=True)
    ap.add_argument("--format", default="ctsv3",
                    help="INCEpTION format id (WebAnno TSV3 = ctsv3)")
    ap.add_argument("--overwrite", action="store_true",
                    help="delete existing same-named doc first")
    args = ap.parse_args()

    auth = HTTPBasicAuth(args.user, args.password)
    api = args.base_url.rstrip("/")
    proj = args.project_id
    docs_ep = f"{api}/api/aero/v1/projects/{proj}/documents"

    r = requests.get(docs_ep, auth=auth)
    if r.status_code == 401:
        sys.exit("401 Unauthorized — check user/password.")
    if r.status_code == 403:
        sys.exit("403 Forbidden — user lacks ROLE_REMOTE.")
    if r.status_code == 404:
        sys.exit(f"404 — project {proj} not found (check --project-id).")
    r.raise_for_status()
    existing = {d["name"] for d in r.json().get("body", [])}
    print(f"Project {proj}: {len(existing)} document(s) already present.")

    tsvs = sorted(glob.glob(os.path.join(args.tsv_dir, "*.tsv")))
    if not tsvs:
        sys.exit(f"No .tsv files in {args.tsv_dir}")
    print(f"Found {len(tsvs)} TSV file(s) to import.")

    for path in tsvs:
        name = os.path.basename(path)
        if name in existing:
            if args.overwrite:
                did = next(d["id"] for d in r.json()["body"] if d["name"] == name)
                requests.delete(f"{docs_ep}/{did}", auth=auth).raise_for_status()
                print(f"  overwrote: deleted existing {name}")
            else:
                print(f"  skip (exists): {name}")
                continue
        with open(path, "rb") as fh:
            files = {"content": (name, fh, "text/plain")}
            data = {"name": name, "format": args.format}
            pr = requests.post(docs_ep, auth=auth, files=files, data=data)
        if pr.status_code not in (200, 201):
            print(f"  FAIL {name}: {pr.status_code} {pr.text[:200]}")
        else:
            print(f"  imported: {name}")

    print("Done.")

if __name__ == "__main__":
    main()
