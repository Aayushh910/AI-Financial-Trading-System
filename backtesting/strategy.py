def apply_risk_filter(
    signals,
    volatility_pred,
    threshold
):

    filtered_signals = []

    for sig, vol in zip(
        signals,
        volatility_pred
    ):

        # Low Volatility
        if vol <= threshold:

            filtered_signals.append(
                sig
            )

        # High Volatility
        else:

            filtered_signals.append(
                0
            )

    return filtered_signals