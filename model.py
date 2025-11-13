from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_model(data, threshold=1.0):
    """
    Train two Random Forest classifiers:
    - model_up: predicts if price will rise ≥ threshold %
    - model_down: predicts if price will fall ≤ −threshold %
    """
    data["Pct_Change"] = data["Close"].pct_change() * 100
    data["Label_Up"] = (data["Pct_Change"].shift(-1) >= threshold).astype(int)
    data["Label_Down"] = (data["Pct_Change"].shift(-1) <= -threshold).astype(int)

    # Drop NaNs from indicators and pct_change
    data = data.dropna(subset=["RSI", "SMA_50", "SMA_200", "MACD", "Label_Up", "Label_Down"])

    features = data[["RSI", "SMA_50", "SMA_200", "MACD"]]
    labels_up = data["Label_Up"]
    labels_down = data["Label_Down"]

    X_train, X_test, y_train_up, y_test_up, y_train_down, y_test_down = train_test_split(
        features, labels_up, labels_down, test_size=0.2, shuffle=False
    )

    model_up = RandomForestClassifier(n_estimators=200, random_state=42)
    model_up.fit(X_train, y_train_up)

    model_down = RandomForestClassifier(n_estimators=200, random_state=42)
    model_down.fit(X_train, y_train_down)

    return model_up, model_down, X_test, y_test_up, y_test_down
