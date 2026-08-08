# dashboard/utils/formatting.py

def format_currency(value):
    """Format float value to USD currency format."""
    if value is None or (isinstance(value, str) and value == "N/A"):
        return "N/A"
    try:
        val = float(value)
        return f"${val:,.2f}"
    except (ValueError, TypeError):
        return str(value)

def format_pct(value, decimals=2, include_sign=False):
    """Format float value to percentage string."""
    if value is None or (isinstance(value, str) and value == "N/A"):
        return "N/A"
    try:
        val = float(value)
        sign = "+" if include_sign and val > 0 else ""
        return f"{sign}{val:.{decimals}f}%"
    except (ValueError, TypeError):
        return str(value)

def format_number(value, decimals=4):
    """Format float value to specified decimal precision."""
    if value is None or (isinstance(value, str) and value == "N/A"):
        return "N/A"
    try:
        val = float(value)
        return f"{val:.{decimals}f}"
    except (ValueError, TypeError):
        return str(value)

def get_signal_label(signal):
    """Return text representation for numerical signal (-1, 0, 1)."""
    if signal == 1 or str(signal).upper() == "BUY":
        return "BUY"
    elif signal == -1 or str(signal).upper() == "SELL":
        return "SELL"
    else:
        return "HOLD"
