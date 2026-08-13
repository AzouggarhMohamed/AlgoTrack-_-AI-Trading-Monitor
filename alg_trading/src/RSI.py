def calcul_rsi(prix,period = 14):
    y = prix.diff()
    gain = y.clip(lower = 0)
    loss = -y.clip(upper = 0)
    avg_gain = gain.rolling(window = period).mean()
    avg_loss = loss.rolling(window = period).mean()
    rs = avg_gain/avg_loss
    rsi = 100-(100/(1+rs))
    return rsi    