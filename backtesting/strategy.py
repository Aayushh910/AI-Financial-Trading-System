def apply_risk_filter(signals, volatility_pred, threshold):
    filtered_signals = []

    for sig, vol in zip(signals, volatility_pred):
        if vol < threshold:
            filtered_signals.append(sig)
        else:
            filtered_signals.append(0)  # no trade

    return filtered_signals