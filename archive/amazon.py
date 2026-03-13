import pandas as pd

df = pd.read_csv(
    "/Users/boyuedong/Desktop/new3:11/full_dataset_cleaned.csv",
    usecols=["STOCK", "DATE"]
)

amazon_df = df[df["STOCK"].astype(str).str.strip().str.lower() == "amazon"].copy()
amazon_df["DATE"] = pd.to_datetime(amazon_df["DATE"], format="%d/%m/%Y", errors="coerce")
amazon_df = amazon_df.dropna(subset=["DATE"])

amazon_dates = (
    amazon_df[["DATE"]]
    .drop_duplicates()
    .sort_values("DATE")
)

amazon_dates["DATE"] = amazon_dates["DATE"].dt.strftime("%Y-%m-%d")
amazon_dates.to_csv("/Users/boyuedong/Desktop/new3:11/amazon_dates.csv", index=False)

print("Saved Amazon dates to amazon_dates.csv", flush=True)