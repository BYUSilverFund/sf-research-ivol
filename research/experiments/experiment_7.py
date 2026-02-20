import datetime as dt
from pathlib import Path

import great_tables as gt
import polars as pl
import sf_quant.data as sfd

# We estimate the correlation between the return streams of IVOL and BaB
start = dt.date(1996, 1, 1)
end = dt.date(2024, 12, 31)
results_folder = Path("results/experiment_7")

# Create results folder
results_folder.mkdir(parents=True, exist_ok=True)

# Load MVO weights
ivol_weights = pl.read_parquet(f"weights/ivol/50/*.parquet")
bab_weights = pl.read_parquet(f"weights/bab/50/*.parquet")


# Get returns
returns = (
    sfd.load_assets(
        start=start, end=end, columns=["date", "barrid", "return"], in_universe=True
    )
    .sort("date", "barrid")
    .select(
        "date",
        "barrid",
        pl.col("return").truediv(100).shift(-1).over("barrid").alias("forward_return"),
    )
)

# Compute portfolio returns
ivol_portfolio_returns = (
    ivol_weights.join(other=returns, on=["date", "barrid"], how="left")
    .group_by("date")
    .agg(pl.col("forward_return").mul(pl.col("weight")).sum().alias("ivol_return"))
    .sort("date")
)

bab_portfolio_returns = (
    bab_weights.join(other=returns, on=["date", "barrid"], how="left")
    .group_by("date")
    .agg(pl.col("forward_return").mul(pl.col("weight")).sum().alias("bab_return"))
    .sort("date")
)

portfolio_returns = (
    ivol_portfolio_returns.join(other=bab_portfolio_returns, on=["date"], how="left")
    .sort("date")
    .select(["date", "bab_return", "ivol_return"])
)


# Create summary table
summary = portfolio_returns.select(
    pl.corr("bab_return", "ivol_return").alias("IVOL_BaB_Correlation")
)

table = (
    gt.GT(summary)
    .tab_header(title="IVOL vs. BaB Full Sample Comparison")
    .cols_label(
        IVOL_BaB_Correlation="Correlation",
    )
    .fmt_number("IVOL_BaB_Correlation", decimals=2)
    .opt_stylize(style=4, color="gray")
)

table_path = results_folder / "correlation_table.png"
table.save(table_path, scale=3)
