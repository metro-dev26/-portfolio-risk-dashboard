"""One-off: fetch Fama-French 3-factor DAILY data and write factors.csv (decimals),
filtered to 2018-01-01+ to align with prices.csv. Run once from repo root:
    python tools/snapshot_factors.py
Regenerate only to refresh the data window."""
import io
import zipfile
import urllib.request
import pandas as pd

try:
    import truststore
    truststore.inject_into_ssl()   # survive HTTPS-inspecting networks
except Exception:
    pass

URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
       "F-F_Research_Data_Factors_daily_CSV.zip")


def main():
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, timeout=30).read()
    z = zipfile.ZipFile(io.BytesIO(raw))
    txt = z.read(z.namelist()[0]).decode("latin-1")
    lines = txt.splitlines()
    hdr = next(i for i, l in enumerate(lines) if l.strip().startswith(",Mkt-RF"))
    rows = []
    for l in lines[hdr + 1:]:
        parts = [p.strip() for p in l.split(",")]
        if len(parts) == 5 and len(parts[0]) == 8 and parts[0].isdigit():
            d = parts[0]
            if d >= "20180101":
                rows.append([d] + [float(x) / 100.0 for x in parts[1:]])  # percent -> decimal
    df = pd.DataFrame(rows, columns=["Date", "Mkt-RF", "SMB", "HML", "RF"])
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
    df = df.set_index("Date").sort_index()
    df.to_csv("factors.csv")
    print(f"wrote factors.csv: {len(df)} rows, {df.index.min().date()} -> {df.index.max().date()}")
    print(df.head(2).to_string())


if __name__ == "__main__":
    main()
