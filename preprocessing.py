import pandas as pd

def preprocess_dates(df, date_columns):
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

def clean_dataframes(products, purchases, sales, clients, suppliers):
    purchases = preprocess_dates(purchases, ["purchase_date"])
    sales     = preprocess_dates(sales,     ["sale_date"])
    for df in [products, purchases, sales, clients, suppliers]:
        df.columns = [c.strip().lower() for c in df.columns]
    purchases = purchases.dropna(subset=["purchase_price"])
    sales     = sales.dropna(subset=["selling_price"])
    return products, purchases, sales, clients, suppliers
