import pandas as pd

def compare_supplier_prices(purchases, product_id):
    rows = purchases[purchases["product_id"] == product_id].sort_values("purchase_date")
    if len(rows) < 2:
        return None
    old_price = rows.iloc[-2]["purchase_price"]
    new_price = rows.iloc[-1]["purchase_price"]
    change    = round(((new_price - old_price) / old_price) * 100, 2)
    return {
        "old_price":      old_price,
        "new_price":      new_price,
        "change_percent": change,
        "direction":      "up" if change > 0 else "down" if change < 0 else "stable",
    }

def all_supplier_changes(purchases):
    """Return price change summary for every product that has ≥2 purchases."""
    results = []
    for pid in purchases["product_id"].unique():
        cmp = compare_supplier_prices(purchases, pid)
        if cmp:
            cmp["product_id"] = pid
            results.append(cmp)
    return pd.DataFrame(results)
