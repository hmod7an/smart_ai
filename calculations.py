import pandas as pd

def merge_data(products, purchases, sales):
    # Use the latest purchase price per product
    latest_purchases = (
        purchases.sort_values("purchase_date")
        .groupby("product_id")
        .last()
        .reset_index()
    )
    merged = sales.merge(products, on="product_id", how="left")
    merged = merged.merge(
        latest_purchases[["product_id", "purchase_price", "tax", "shipping", "expenses", "supplier_name"]],
        on="product_id", how="left"
    )
    return merged

def calculate_total_cost(row):
    return row["purchase_price"] + row["tax"] + row["shipping"] + row["expenses"]

def calculate_profit(row):
    total_cost = calculate_total_cost(row)
    return (row["selling_price"] - total_cost) * row["quantity"]

def calculate_margin(row):
    total_revenue = row["selling_price"] * row["quantity"]
    total_cost    = calculate_total_cost(row) * row["quantity"]
    if total_revenue == 0:
        return 0
    return round(((total_revenue - total_cost) / total_revenue) * 100, 2)

def build_merged_df(products, purchases, sales):
    merged = merge_data(products, purchases, sales)
    merged["total_cost"] = merged.apply(calculate_total_cost, axis=1)
    merged["profit"]     = merged.apply(calculate_profit,    axis=1)
    merged["margin"]     = merged.apply(calculate_margin,    axis=1)
    return merged
