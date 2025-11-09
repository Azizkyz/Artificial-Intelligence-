import math
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SEED = 42
rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------
# Laplace smoothing helper
# ---------------------------------------------------------------
def laplace_smooth(count, total, num_categories, alpha=1.0):
    """Avoid zero probabilities with Laplace smoothing."""
    return (count + alpha) / (total + alpha * num_categories) if total > 0 else 1.0 / num_categories


# ---------------------------------------------------------------
# Naïve Bayes Classifier for categorical data
# ---------------------------------------------------------------
class CategoricalNaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.classes = None
        self.feature_values = {}
        self.priors = {}
        self.likelihoods = {}

    def fit(self, X, y):
        self.classes = sorted(y.unique())
        self.feature_values = {col: sorted(X[col].unique()) for col in X.columns}
        self.priors = {cls: (y == cls).mean() for cls in self.classes}
        self.likelihoods = {col: {} for col in X.columns}

        for col in X.columns:
            num_vals = len(self.feature_values[col])
            for cls in self.classes:
                subset = X[y == cls]
                total = len(subset)
                self.likelihoods[col][cls] = {}
                for val in self.feature_values[col]:
                    count = (subset[col] == val).sum()
                    self.likelihoods[col][cls][val] = laplace_smooth(count, total, num_vals, self.alpha)
        return self

    def predict_proba_row(self, row):
        """Return normalized probabilities for one sample."""
        log_probs = {}
        for cls in self.classes:
            log_prob = math.log(self.priors[cls] + 1e-15)
            for feature, value in row.items():
                num_vals = len(self.feature_values[feature])
                prob = self.likelihoods[feature][cls].get(value, 1.0 / num_vals)
                log_prob += math.log(prob + 1e-15)
            log_probs[cls] = log_prob

        max_log = max(log_probs.values())
        exp_probs = {cls: math.exp(log_probs[cls] - max_log) for cls in self.classes}
        total = sum(exp_probs.values())
        return {cls: exp_probs[cls] / total for cls in self.classes}

    def predict(self, X):
        """Predict class for each observation."""
        return [
            max(self.predict_proba_row(row).items(), key=lambda kv: kv[1])[0]
            for _, row in X.iterrows()
        ]


# ---------------------------------------------------------------
# Dataset generation with strong correlations
# ---------------------------------------------------------------
def create_high_accuracy_weather_dataset(path="weather_high_accuracy.csv", n=100_000):
    np.random.seed(SEED)
    outlooks = ["Sunny", "Overcast", "Rain"]
    temps = ["Hot", "Mild", "Cool"]
    humidity = ["High", "Low"]
    wind = ["Strong", "Weak"]
    cloudy = ["Yes", "No"]

    data = {
        "Outlook": np.random.choice(outlooks, n),
        "Temperature": np.random.choice(temps, n),
        "Humidity": np.random.choice(humidity, n),
        "Wind": np.random.choice(wind, n),
        "Cloudy": np.random.choice(cloudy, n)
    }
    df = pd.DataFrame(data)

    probs = []
    for _, row in df.iterrows():
        prob = 0.05
        if row["Cloudy"] == "Yes":
            prob += 0.45
        if row["Humidity"] == "High":
            prob += 0.35
        if row["Outlook"] == "Rain":
            prob += 0.35
        if row["Temperature"] == "Cool":
            prob += 0.15
        if row["Wind"] == "Strong":
            prob -= 0.20
        if row["Outlook"] == "Sunny":
            prob -= 0.25
        prob = np.clip(prob + np.random.normal(0, 0.02), 0.02, 0.98)
        probs.append(prob)

    df["Rain"] = ["Yes" if np.random.rand() < p else "No" for p in probs]
    df.to_csv(path, index=False)
    print(f"✅ Generated dataset → {path}")
    return df


# ---------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------
def main():
    # Step 1 — Dataset
    df = create_high_accuracy_weather_dataset()
    target_col = "Rain"
    features = ["Outlook", "Temperature", "Humidity", "Wind", "Cloudy"]

    # Step 2 — Split 70/30
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    n_train = int(len(df) * 0.7)
    train = df.iloc[:n_train]
    test = df.iloc[n_train:]
    X_train, y_train = train[features], train[target_col]
    X_test, y_test = test[features], test[target_col]

    # Step 3 — Train
    nb = CategoricalNaiveBayes(alpha=1.0).fit(X_train, y_train)

    # Step 4 — Predictions
    y_pred_test = nb.predict(X_test)
    y_pred_train = nb.predict(X_train)

    # Accuracy
    train_acc = (pd.Series(y_pred_train).values == y_train.values).mean()
    test_acc = (pd.Series(y_pred_test).values == y_test.values).mean()

    # Step 5 — Confusion Matrix components
    y_true = np.array(y_test)
    y_pred = np.array(y_pred_test)
    TP = np.sum((y_true == "Yes") & (y_pred == "Yes"))
    TN = np.sum((y_true == "No") & (y_pred == "No"))
    FP = np.sum((y_true == "No") & (y_pred == "Yes"))
    FN = np.sum((y_true == "Yes") & (y_pred == "No"))

    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

    print("\n--- Model Performance ---")
    print(f"Training Accuracy : {train_acc:.4f}")
    print(f"Testing Accuracy  : {test_acc:.4f}")
    print(f"True Positives   : {TP}")
    print(f"True Negatives   : {TN}")
    print(f"False Positives  : {FP}")
    print(f"False Negatives  : {FN}")
    print(f"Sensitivity (TPR): {sensitivity:.4f}")
    print(f"Specificity (TNR): {specificity:.4f}")

    # Example query
    query = pd.Series({
        "Outlook": "Rain",
        "Temperature": "Mild",
        "Humidity": "High",
        "Wind": "Weak",
        "Cloudy": "Yes"
    })
    q_probs = nb.predict_proba_row(query)
    q_pred = max(q_probs.items(), key=lambda kv: kv[1])[0]

    print("\n--- Example Query ---")
    print("Conditions:", dict(query))
    print("Predicted Class:", q_pred)
    print("Posterior Probabilities:", {k: round(v, 4) for k, v in q_probs.items()})

    # Visualization
    combo = (
        df.assign(RainYes=(df[target_col] == "Yes").astype(int))
        .groupby(["Humidity", "Cloudy"])["RainYes"].mean()
        .reset_index()
        .rename(columns={"RainYes": "P_Rain_Yes"})
    )

    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(combo["P_Rain_Yes"])
    bars = ax.bar(combo.index, combo["P_Rain_Yes"], color=colors, edgecolor="black")

    ax.set_xticks(combo.index)
    ax.set_xticklabels(combo["Humidity"] + " / " + combo["Cloudy"], rotation=30)
    ax.set_title("Probability of Rain by Humidity & Cloudiness", fontsize=13, weight="bold")
    ax.set_xlabel("Conditions (Humidity / Cloudy)")
    ax.set_ylabel("P(Rain = Yes)")
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--", alpha=0.6)

    for i, bar in enumerate(bars):
        val = combo["P_Rain_Yes"].iloc[i]
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val:.2f}",
                ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig("rain_probability_sensitivity_specificity.png", dpi=180)
    plt.show()
    print("\nSaved visualization → rain_probability_sensitivity_specificity.png")

    # Save model
    model_data = {
        "alpha": nb.alpha,
        "classes": nb.classes,
        "priors": nb.priors,
        "feature_values": nb.feature_values,
        "likelihoods": nb.likelihoods
    }
    with open("naive_bayes_sensitivity_specificity_model.json", "w", encoding="utf-8") as f:
        json.dump(model_data, f, indent=2)
    print("Model saved → naive_bayes_sensitivity_specificity_model.json")


if __name__ == "__main__":
    main()
