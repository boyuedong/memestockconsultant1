import re
import pandas as pd
from pathlib import Path

# =========================
# File paths
# =========================

file_path = Path("/Users/boyuedong/Downloads/tweet_sentiment/full_dataset-release.csv")
output_csv = Path("/Users/boyuedong/Desktop/new3:11/full_dataset_cleaned.csv")
bad_rows_file = Path("/Users/boyuedong/Desktop/new3:11/bad_rows_preview.txt")

# =========================
# Final column names
# =========================
columns = [
    "ROW_ID",
    "TWEET",
    "STOCK",
    "DATE",
    "LAST_PRICE",
    "1_DAY_RETURN",
    "2_DAY_RETURN",
    "3_DAY_RETURN",
    "7_DAY_RETURN",
    "PX_VOLUME",
    "VOLATILITY_10D",
    "VOLATILITY_30D",
    "LSTM_POLARITY",
    "TEXTBLOB_POLARITY"
]

numeric_cols = [
    "LAST_PRICE",
    "1_DAY_RETURN",
    "2_DAY_RETURN",
    "3_DAY_RETURN",
    "7_DAY_RETURN",
    "PX_VOLUME",
    "VOLATILITY_10D",
    "VOLATILITY_30D",
    "LSTM_POLARITY",
    "TEXTBLOB_POLARITY"
]

date_pat = re.compile(r"\d{2}/\d{2}/\d{4}")

print("Reading file...")
with open(file_path, "r", encoding="utf-8", errors="replace") as f:
    lines = [line.rstrip("\n") for line in f]

print(f"Read {len(lines):,} lines. Rebuilding rows...")

# Skip header
data_lines = lines[1:]

# =========================
# Step 1: Rebuild logical rows
# =========================
records = []
current = None
expected_next_id = None

for i, line in enumerate(data_lines, start=1):
    m = re.match(r"^(\d+),(.*)$", line)

    if m:
        candidate_id = int(m.group(1))

        if current is None:
            current = line
            expected_next_id = candidate_id + 1
        elif candidate_id == expected_next_id:
            records.append(current)
            current = line
            expected_next_id = candidate_id + 1
        else:
            # Looks like digits+comma, but is not the next real row id
            current += " " + line.strip()
    else:
        if current is not None:
            current += " " + line.strip()

    if i % 50000 == 0:
        print(f"  Processed {i:,} / {len(data_lines):,} lines...")

if current is not None:
    records.append(current)

print(f"Built {len(records):,} records. Parsing...")

# =========================
# Step 2: Parse each rebuilt row
# =========================
rows = []
bad_records = []

for rec in records:
    try:
        row_id, rest = rec.split(",", 1)

        matches = list(date_pat.finditer(rest))
        if not matches:
            bad_records.append(rec)
            continue

        # Use the LAST date in the row as the DATE column
        m = matches[-1]
        date_start, date_end = m.span()
        date_val = m.group()

        left = rest[:date_start]
        right = rest[date_end:]

        # left should end with: ...,STOCK,
        left_parts = left.rsplit(",", 2)
        if len(left_parts) != 3:
            bad_records.append(rec)
            continue

        tweet = left_parts[0].strip().strip('"')
        stock = left_parts[1].strip()

        # right should begin with comma, then 10 remaining fields
        right = right.lstrip(",")
        tail = right.split(",")

        if len(tail) != 10:
            bad_records.append(rec)
            continue

        tweet = " ".join(tweet.split())
        row = [row_id, tweet, stock, date_val] + [x.strip() for x in tail]
        rows.append(row)

    except Exception as e:
        bad_records.append(f"{repr(rec)}\nEXCEPTION: {repr(e)}")

# =========================
# Step 3: Build DataFrame
# =========================
df = pd.DataFrame(rows, columns=columns)

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["TWEET"] = df["TWEET"].astype(str).str.strip().str.strip('"')

# =========================
# Step 4: Save outputs
# =========================
df.to_csv(output_csv, index=False)

with open(bad_rows_file, "w", encoding="utf-8") as f:
    f.write(f"Total bad rows: {len(bad_records)}\n")
    for i, rec in enumerate(bad_records[:500]):
        f.write(f"\n--- BAD ROW {i} ---\n")
        f.write(str(rec))
        f.write("\n")

# =========================
# Step 5: Summary
# =========================
print("Done.")
print("DataFrame shape:", df.shape)
print("Bad rows skipped:", len(bad_records))
print(f"Cleaned CSV saved to: {output_csv}")
print(f"Bad row previews saved to: {bad_rows_file}")
print(df.head())