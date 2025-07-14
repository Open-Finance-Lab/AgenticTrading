import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = "tests/data"
SOURCES = ["nasdaq", "polygon", "yfinance"]
COLUMNS = ["Close", "Volume", "Open", "High", "Low"]
OUTPUT_DIR = "tests/data_error_rate"

def load_data(source, ticker):
    path = os.path.join(BASE_DIR, source, f"{ticker}.csv")
    df = pd.read_csv(path)

    df = df.rename(columns={"Close/Last": "Close"})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    for col in COLUMNS:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(r"[\$,]", "", regex=True)
            .replace("null", pd.NA)
            .astype(float)
        )

    return df[["Date"] + COLUMNS]

def compare_data(ticker):
    try:
        dfs = {src: load_data(src, ticker) for src in SOURCES}
    except FileNotFoundError:
        print(f"Missing file for {ticker}, skipping.")
        return None

    base_df = dfs["nasdaq"]
    results = []

    for source in ["polygon", "yfinance"]:
        merged = pd.merge(base_df, dfs[source], on="Date", suffixes=("_nasdaq", f"_{source}"))
        for col in COLUMNS:
            try:
                abs_err = (merged[f"{col}_{source}"] - merged[f"{col}_nasdaq"]).abs()
                pct_err = abs_err / merged[f"{col}_nasdaq"].replace(0, pd.NA).abs() * 100
            except Exception as e:
                print(f"Error comparing {ticker} {col} for {source}: {e}")
                continue

            for i in range(len(merged)):
                results.append({
                    "Ticker": ticker,
                    "Date": merged.loc[i, "Date"],
                    "Source": source,
                    "Column": col,
                    "NASDAQ": merged.loc[i, f"{col}_nasdaq"],
                    "SourceVal": merged.loc[i, f"{col}_{source}"],
                    "AbsError": abs_err.iloc[i],
                    "PctError": pct_err.iloc[i],
                })

    return pd.DataFrame(results)

def generate_heatmap(df):
    summary = df.groupby(["Ticker", "Source", "Column"])["PctError"].mean().reset_index()
    heatmap_df = summary.pivot_table(index="Ticker", columns=["Source", "Column"], values="PctError")

    plt.figure(figsize=(16, 8))
    sns.heatmap(heatmap_df, annot=True, fmt=".2f", cmap="Reds", cbar_kws={'label': '% Error'})
    plt.title("Average % Error Compared to NASDAQ")
    plt.tight_layout()

    heatmap_path = os.path.join(OUTPUT_DIR, "error_heatmap.png")
    plt.savefig(heatmap_path)
    print(f"Error heatmap saved to: {heatmap_path}")
    plt.close()

def main():
    tickers = [
        f[:-4] for f in os.listdir(os.path.join(BASE_DIR, "nasdaq"))
        if f.endswith(".csv")
    ]
    all_results = []

    for ticker in tickers:
        df_result = compare_data(ticker)
        if df_result is not None:
            all_results.append(df_result)

    if not all_results:
        print("No results to write.")
        return

    full_df = pd.concat(all_results, ignore_index=True)
    log_path = os.path.join(OUTPUT_DIR, "comparison_log.csv")
    full_df.to_csv(log_path, index=False)
    print(f"Comparison log written to: {log_path}")

    generate_heatmap(full_df)

if __name__ == "__main__":
    main()
