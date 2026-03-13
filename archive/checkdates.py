import pandas as pd

def out(msg):
    print(msg, flush=True)

out("Loading CSV...")
df = pd.read_csv("/Users/boyuedong/Desktop/new3:11/full_dataset_cleaned.csv")
out(f"Loaded {len(df):,} rows.")

# Parse DATE column
df["DATE"] = pd.to_datetime(df["DATE"], format="%d/%m/%Y", errors="coerce")

# Check for bad/unparsed dates
bad_dates = df[df["DATE"].isna()]
out("Number of invalid dates: " + str(len(bad_dates)))

# Drop invalid ones for date-range checking
valid_dates = df["DATE"].dropna()

out("Min date: " + str(valid_dates.min()))
out("Max date: " + str(valid_dates.max()))

if len(valid_dates) == 0:
    out("No valid dates to check. Exiting.")
    raise SystemExit(0)

# Build expected full-year range
year = valid_dates.dt.year.mode()[0]   # most common year in the dataset
expected = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq="D")

# Get actual unique dates
actual = pd.DatetimeIndex(valid_dates.dt.normalize().unique()).sort_values()

# Find missing dates
missing = expected.difference(actual)

out("Expected number of dates: " + str(len(expected)))
out("Actual unique dates: " + str(len(actual)))
out("Number of missing dates: " + str(len(missing)))

if len(missing) == 0:
    out("All dates from Jan 1 to Dec 31 are present.")
else:
    out("Missing dates:")
    for d in missing:
        out(d.strftime("%Y-%m-%d"))