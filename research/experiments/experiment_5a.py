# BaB + IVOL

import datetime as dt
from pathlib import Path

import polars as pl
import sf_quant.data as sfd
import sf_quant.performance as sfp
from dotenv import load_dotenv

from research.utils import run_backtest_parallel

# Load environment variables
load_dotenv()

# Parameters
start = dt.date(1996, 1, 1)
end = dt.date(2024, 12, 31)
price_filter = 5
signal_name = "ivol_bab"
IC = 0.05
gamma = 50
n_cpus = 8
constraints = ["ZeroBeta", "ZeroInvestment"]
results_folder = Path("results/experiment_5")

# Create results folder
results_folder.mkdir(parents=True, exist_ok=True)

# Get data
data = sfd.load_assets(
    start=start,
    end=end,
    columns=[
        "date",
        "barrid",
        "ticker",
        "price",
        "return",
        "specific_return",
        "specific_risk",
        "predicted_beta",
    ],
    in_universe=True,
).with_columns(
    pl.col("return").truediv(100),
    pl.col("specific_return").truediv(100),
    pl.col("specific_risk").truediv(100),
)

# compute signal
signals = data.sort("barrid", "date").with_columns(
    pl.col("predicted_beta").mul(-1).shift(1).over("barrid").alias("bab"),
    pl.col("specific_risk").mul(-1).shift(1).over("barrid").alias("ivol"),
)

# Filter universe
filtered = signals.filter(
    pl.col("price").shift(1).over("barrid").gt(price_filter),
    pl.col("ivol").is_not_null(),
    pl.col("bab").is_not_null(),
    pl.col("predicted_beta").is_not_null(),
    pl.col("specific_risk").is_not_null(),
)

signals = ["bab", "ivol"]

# Compute scores
scores = filtered.select(
    "date",
    "barrid",
    "predicted_beta",
    "specific_risk",
    pl.col("bab")
    .sub(pl.col("bab").mean())
    .truediv(pl.col("bab").std())
    .over("date")
    .alias("bab_score"),
    pl.col("ivol")
    .sub(pl.col("ivol").mean())
    .truediv(pl.col("ivol").std())
    .over("date")
    .alias("ivol_score"),
)

# Compute alphas
alphas = (
    scores.with_columns(
        pl.col("bab_score").mul(IC).mul("specific_risk").alias("bab_alpha"),
        pl.col("ivol_score").mul(IC).mul("specific_risk").alias("ivol_alpha"),
    )
    .select("date", "barrid", "bab_alpha", "ivol_alpha", "predicted_beta")
    .sort("date", "barrid")
)

# compute combined alphas
alphas = alphas.with_columns(
    pl.mean_horizontal([pl.col(f"{s}_alpha") for s in signals]).alias("alpha")
).sort(["barrid", "date"])


# Get forward returns
forward_returns = (
    data.sort("date", "barrid")
    .select(
        "date", "barrid", pl.col("return").shift(-1).over("barrid").alias("fwd_return")
    )
    .drop_nulls("fwd_return")
)

# Merge alphas and forward returns
merged = alphas.join(other=forward_returns, on=["date", "barrid"], how="inner")

# Get merged alphas and forward returns (inner join)
merged_alphas = merged.select("date", "barrid", "alpha")
merged_forward_returns = merged.select("date", "barrid", "fwd_return")

# Get ics
ics = sfp.generate_alpha_ics(
    alphas=alphas, rets=forward_returns, method="rank", window=22
)

# Save ic chart
rank_chart_path = results_folder / "rank_ic_chart.png"
pearson_chart_path = results_folder / "pearson_ic_chart.png"
sfp.generate_ic_chart(
    ics=ics,
    title=f"{signal_name} Cumulative IC",
    ic_type="Rank",
    file_name=rank_chart_path,
)
sfp.generate_ic_chart(
    ics=ics,
    title=f"{signal_name} Cumulative IC",
    ic_type="Pearson",
    file_name=pearson_chart_path,
)

# Run parallelized backtest
run_backtest_parallel(
    data=alphas,
    signal_name=signal_name,
    constraints=constraints,
    gamma=gamma,
    n_cpus=n_cpus,
)
