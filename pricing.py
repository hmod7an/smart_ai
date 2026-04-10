def suggest_selling_price(total_cost, target_margin_percent):
    """
    Given a total cost and a desired profit margin (%),
    returns the minimum selling price to achieve that margin.
    Formula: price = cost / (1 - margin/100)
    """
    if target_margin_percent >= 100:
        raise ValueError("Margin must be less than 100%")
    return round(total_cost / (1 - target_margin_percent / 100), 2)

def price_sensitivity(total_cost, margins=None):
    """Return a dict of {margin: suggested_price} for a range of margins."""
    if margins is None:
        margins = [10, 15, 20, 25, 30, 40, 50]
    return {m: suggest_selling_price(total_cost, m) for m in margins}
