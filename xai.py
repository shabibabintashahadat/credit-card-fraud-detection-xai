import os
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import torch
import torch.nn as nn

# Ensure results folder exists
os.makedirs("results", exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 12})

print("Explainable AI (XAI) Pipeline")


# 1. REDEFINING THE PYTORCH ARCHITECTURE
class DNNClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.network(x)


# 2. LOAD SAVED MODEL, SCALER, AND DATA
model_path = os.path.join("results", "dnn_model.pt")
print(f"Loading model and scaler from {model_path}...")
checkpoint = torch.load(model_path, weights_only=False)
scaler = checkpoint["scaler"]
input_dim = checkpoint["model_config"]["input_dim"]

model = DNNClassifier(input_dim)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Loading dataset...")
dataset_path = os.path.join("dataset", "creditcard_2023.csv")
df = pd.read_csv(dataset_path)
if "id" in df.columns:
    df = df.drop("id", axis=1)

X = df.drop("Class", axis=1)
y = df["Class"]
feature_names = X.columns.tolist()

df_sample = df.sample(n=5000, random_state=42)
X_sample = df_sample.drop("Class", axis=1)
y_sample = df_sample["Class"]

X_scaled = scaler.transform(X_sample)
X_tensor = torch.FloatTensor(X_scaled)


# 3. WRAPPER FUNCTION FOR LIME AND SHAP
def predict_proba_wrapper(numpy_data):
    tensor_data = torch.FloatTensor(numpy_data)
    with torch.no_grad():
        fraud_probs = model(tensor_data).numpy().flatten()

    legit_probs = 1.0 - fraud_probs
    return np.vstack((legit_probs, fraud_probs)).T


# 4. SHAP
print("\nGenerating SHAP Explanations")

background_data = shap.sample(X_scaled, 100)
test_data = shap.sample(X_scaled, 200)

explainer_shap = shap.KernelExplainer(
    lambda x: predict_proba_wrapper(x)[:, 1],
    background_data,
)

shap_values = explainer_shap.shap_values(test_data)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, test_data, feature_names=feature_names, show=False)
plt.title("SHAP Summary Plot: Global Feature Importance")
plt.tight_layout()
plt.savefig("results\\xai_shap_summary.png")
print("Saved SHAP plot to results\\xai_shap_summary.png")
plt.clf()


# 5. LIME
print("\nGenerating LIME Explanation for a single Fraudulent transaction")

fraud_indices = np.where(y_sample == 1)[0]
sample_idx = fraud_indices[0]
sample_transaction_scaled = X_scaled[sample_idx]

explainer_lime = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_scaled,
    feature_names=feature_names,
    class_names=["Legitimate", "Fraud"],
    mode="classification",
    random_state=42,
)

exp = explainer_lime.explain_instance(
    data_row=sample_transaction_scaled,
    predict_fn=predict_proba_wrapper,
    num_features=6,
)

fig = exp.as_pyplot_figure()
plt.title(f"LIME Explanation for Fraudulent Transaction (ID: {sample_idx})", y=1.05)
plt.tight_layout()
plt.savefig("results\\xai_lime_explanation.png")
print("Saved LIME plot to results\\xai_lime_explanation.png")

print("\nXAI Complete.")
