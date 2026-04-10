import pandas as pd

def client_profit_report(merged_df):
    report = (
        merged_df.groupby("client_name")
        .agg(
            total_profit=("profit", "sum"),
            total_revenue=("selling_price", lambda x: (x * merged_df.loc[x.index, "quantity"]).sum()),
            num_transactions=("sale_id", "count"),
        )
        .reset_index()
        .sort_values("total_profit", ascending=False)
    )
    report["avg_margin"] = round(
        (report["total_profit"] / report["total_revenue"]) * 100, 2
    )
    return report

def best_client(merged_df):
    report = client_profit_report(merged_df)
    return report.iloc[0] if not report.empty else None
