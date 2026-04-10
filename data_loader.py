import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

def load_data():
    products  = pd.read_csv(os.path.join(DATA_DIR, "products.csv"))
    purchases = pd.read_csv(os.path.join(DATA_DIR, "purchases.csv"))
    sales     = pd.read_csv(os.path.join(DATA_DIR, "sales.csv"))
    clients   = pd.read_csv(os.path.join(DATA_DIR, "clients.csv"))
    suppliers = pd.read_csv(os.path.join(DATA_DIR, "suppliers.csv"))
    return products, purchases, sales, clients, suppliers
