import math
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk

SEED = 42


# ---------------------------------------------------------------
# Laplace smoothing helper
# ---------------------------------------------------------------
def laplace_smooth(count, total, num_categories, alpha=1.0):
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
                    self.likelihoods[col][cls][val] = laplace_smooth(
                        count, total, num_vals, self.alpha
                    )
        return self

    def predict_proba_row(self, row):
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
        return [
            max(self.predict_proba_row(row).items(), key=lambda kv: kv[1])[0]
            for _, row in X.iterrows()
        ]


# ---------------------------------------------------------------
# Tkinter GUI Input Panel
# ---------------------------------------------------------------
def open_input_panel(predict_callback):
    window = tk.Tk()
    window.title("Weather Prediction Input")
    window.geometry("300x350")
    window.resizable(False, False)

    outlooks = ["Sunny", "Overcast", "Rain"]
    temps = ["Hot", "Mild", "Cool"]
    humidity = ["High", "Low"]
    wind = ["Strong", "Weak"]
    cloudy = ["Yes", "No"]

    var_outlook = tk.StringVar(value="Sunny")
    var_temp = tk.StringVar(value="Mild")
    var_humidity = tk.StringVar(value="High")
    var_wind = tk.StringVar(value="Weak")
    var_cloudy = tk.StringVar(value="No")

    ttk.Label(window, text="Select Conditions", font=("Arial", 12, "bold")).pack(pady=10)

    def add_dropdown(label, variable, options):
        ttk.Label(window, text=label).pack()
        ttk.Combobox(window, textvariable=variable, values=options, state="readonly").pack(pady=5)

    add_dropdown("Outlook:", var_outlook, outlooks)
    add_dropdown("Temperature:", var_temp, temps)
    add_dropdown("Humidity:", var_humidity, humidity)
    add_dropdown("Wind:", var_wind, wind)
    add_dropdown("Cloudy:", var_cloudy, cloudy)

    def on_predict():
        conditions = {
            "Outlook": var_outlook.get(),
            "Temperature": var_temp.get(),
            "Humidity": var_humidity.get(),
            "Wind": var_wind.get(),
            "Cloudy": var_cloudy.get()
        }
        predict_callback(conditions)

    ttk.Button(window, text="Predict Rain", command=on_predict).pack(pady=20)
    window.mainloop()


# ---------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------
def main():

    print("\n📂 Loading dataset weather.csv ...")
    df = pd.read_csv("weather.csv")
    print("✅ Dataset loaded successfully!")

    target_col = "Rain"
    features = ["Outlook", "Temperature", "Humidity", "Wind", "Cloudy"]

    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    split = int(len(df) * 0.7)
    train, test = df[:split], df[split:]

    X_train, y_train = train[features], train[target_col]
    X_test, y_test = test[features], test[target_col]

    print("🧠 Training Naïve Bayes model...")
    nb = CategoricalNaiveBayes(alpha=1.0).fit(X_train, y_train)

    print("\n--- Model Performance ---")
    print(f"Training Accuracy: {(nb.predict(X_train) == y_train.values).mean():.4f}")
    print(f"Testing Accuracy : {(nb.predict(X_test) == y_test.values).mean():.4f}")

    # -------------------------------------------------------
    # Compute conditional probabilities for 4 combinations
    # -------------------------------------------------------
    df["RainYes"] = (df["Rain"] == "Yes").astype(int)

    combo = (
        df.groupby(["Humidity", "Cloudy"])["RainYes"]
        .mean()
        .reset_index()
    )

    combo_labels = combo["Humidity"] + " / " + combo["Cloudy"]
    combo_values = combo["RainYes"]

    # -------------------------------------------------------
    # DISPLAY ORIGINAL 4-BAR PLOT
    # -------------------------------------------------------
    plt.figure(figsize=(9, 5))
    plt.bar(combo_labels, combo_values, color=["#4E79A7", "#F28E2B", "#59A14F", "#EDC948"])

    plt.xticks(rotation=30, ha="right")
    plt.ylabel("P(Rain = Yes)")
    plt.title("Probability of Rain by Humidity & Cloudiness")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------
    # GUI CALLBACK
    # -------------------------------------------------------
    def handle_gui_input(conditions):
        print("\n--- USER INPUT RECEIVED FROM GUI ---")
        print("Conditions:", conditions)

        query = pd.Series(conditions)
        q_probs = nb.predict_proba_row(query)
        q_pred = max(q_probs.items(), key=lambda kv: kv[1])[0]

        print(f"\n🌧️ Prediction → Rain = {q_pred}")
        print("Posterior Probabilities:", {k: round(v, 4) for k, v in q_probs.items()})

        # NEW: CLEAR DECISION MESSAGE
        if q_pred == "Yes":
            print("\n🌧️ FINAL DECISION: It WILL RAIN.\n")
        else:
            print("\n☀️ FINAL DECISION: It will NOT rain.\n")

    open_input_panel(handle_gui_input)

    print("\nProgram finished.")


if __name__ == "__main__":
    main()
