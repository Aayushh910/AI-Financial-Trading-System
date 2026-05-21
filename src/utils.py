import matplotlib.pyplot as plt

def plot_equity_curve(data):

    plt.figure(figsize=(12,6))

    plt.plot(
        data['Equity_Curve']
    )

    plt.title(
        "Trading Strategy Equity Curve"
    )

    plt.xlabel("Date")

    plt.ylabel("Portfolio Value")

    plt.savefig(
        "outputs/equity_curve.png"
    )

    plt.show()