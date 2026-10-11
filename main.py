# pyrefly: ignore [missing-import]
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Ensure results folder exists
os.makedirs("results", exist_ok=True)

# 1. Load the dataset
dataset_path = os.path.join("dataset", "creditcard_2023.csv")
if not os.path.exists(dataset_path):
    print(f"Error: Dataset not found at '{dataset_path}'.")
    print("Please download 'creditcard_2023.csv' from Kaggle and place it in the dataset/ folder.")
    print("See dataset/README.md for download instructions.")
    sys.exit(1)

df = pd.read_csv(dataset_path)

# 2. Basic Dataset Info
print("Dataset Shape:", df.shape)
print("Dataset Column Name:", df.columns)
print("\nClass Distribution:")
print(df["Class"].value_counts(normalize=True) * 100)

# 3. Drop the 'id' column
if "id" in df.columns:
    df = df.drop("id", axis=1)

# 4. Scale the 'Amount' column
scaler = RobustScaler()
df["Amount"] = scaler.fit_transform(df["Amount"].values.reshape(-1, 1))
print("\nPreprocessing complete. Data is ready for modeling.")

# Set the aesthetic style of the plots
sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 12})


# ANALYSIS 1: Correlation Matrix
plt.figure(figsize=(12, 8))
corr = df.corr()
corr_with_class = corr["Class"].drop("Class").sort_values()
corr_with_class.plot(
    kind="bar", color=np.where(corr_with_class > 0, "#e74c3c", "#3498db")
)
plt.title("Correlation of Anonymized Features (V1-V28) and Amount with Fraud (Class)")
plt.ylabel("Pearson Correlation Coefficient")
plt.xlabel("Features")
plt.tight_layout()
plt.savefig(os.path.join("results", "dataset_feature_correlation_matrix.png"))

print("Top 3 features positively correlated with fraud:")
print(corr_with_class.tail(3))
print("\nTop 3 features negatively correlated with fraud:")
print(corr_with_class.head(3))



# ANALYSIS 2: Transaction Amount Comparison
plt.figure(figsize=(10, 6))
sns.boxplot(x="Class", y="Amount", data=df, showfliers=False, palette="Set2")
plt.title("Transaction Amount Distribution: Legitimate (0) vs Fraudulent (1)")
plt.xticks([0, 1], ["Legitimate", "Fraudulent"])
plt.ylabel("Transaction Amount")
plt.savefig(os.path.join("results", "dataset_transaction_boxplot.png"))


# ANALYSIS 3: Feature Distribution (Density Plots)

features_to_plot = ["V11", "V4"]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for i, feature in enumerate(features_to_plot):
    sns.kdeplot(
        data=df[df["Class"] == 0][feature],
        label="Legitimate",
        fill=True,
        ax=axes[i],
        color="blue",
    )
    sns.kdeplot(    
        data=df[df["Class"] == 1][feature],
        label="Fraudulent",
        fill=True,
        ax=axes[i],
        color="red",
    )
    axes[i].set_title(f"Distribution of Feature {feature}")
    axes[i].legend()

plt.tight_layout()
plt.savefig(os.path.join("results", "dataset_feature_distribution.png"))


# Sample the data to make plotting faster and clearer
df_sample = df.sample(n=10000, random_state=42)

# ANALYSIS 4: Bivariate Interactions (Top 3 Features)
print("Generating Pairplot...")
top_features = ["V2", "V4", "V11"]

pair_plot = sns.pairplot(
    df_sample,
    vars=top_features,
    hue="Class",
    palette={0: "#3498db", 1: "#e74c3c"},
    plot_kws={"alpha": 0.5, "s": 15},
    diag_kind="kde",
)
pair_plot.fig.suptitle("Bivariate Interactions of Top Correlated Features", y=1.02)
plt.savefig(os.path.join("results", "dataset_feature_interaction(Bivariate_Interactions).png"))

# ANALYSIS 5: t-SNE Manifold Visualization
print("Generating t-SNE Visualization")
X_sample = df_sample.drop("Class", axis=1)
y_sample = df_sample["Class"]
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
X_tsne = tsne.fit_transform(X_sample)

plt.figure(figsize=(10, 8))
sns.scatterplot(
    x=X_tsne[:, 0],
    y=X_tsne[:, 1],
    hue=y_sample,
    palette={0: "#3498db", 1: "#e74c3c"},
    alpha=0.6,
    s=20,
)
plt.title("t-SNE Projection of Transaction Data (10,000 sample)")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.legend(title="Class", labels=["Legitimate (0)", "Fraudulent (1)"])
plt.savefig(os.path.join("results", "dataset_feature_t-SNE_visualization.png"))


# Training Basic Model Logistic Regression and Random Forest

# Split features (X) and target (y)
X = df.drop("Class", axis=1)
y = df["Class"]

# 2. Train-Test Split
# 80% of the data to train and 20% to test.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}\n")

#Shabiba Binta Shahadat (22101907)
# MODEL 1: Logistic Regression
print("--- Training Logistic Regression ---")
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)

report = classification_report(y_test, lr_predictions)
print("Logistic Regression Results:")
print(report)
with open("results\\logistic_regression_report.txt", "w") as f:
    f.write("Logistic Regression Results:\n")
    f.write(report)

cm = confusion_matrix(y_test, lr_predictions)
plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    xticklabels=["Legitimate", "Fraud"],
    yticklabels=["Legitimate", "Fraud"],
)
plt.title("Confusion Matrix - Logistic Regression")
plt.ylabel("Actual Class")
plt.xlabel("Predicted Class")
plt.savefig(os.path.join("results", "logistic_regression_confusion_matrix.png"))

# MODEL 2: Random Forest
print("\n--- Training Random Forest ---")
rf_model = RandomForestClassifier(
    n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

report = classification_report(y_test, rf_predictions)
print("Random Forest Results:")
print(report)
with open(os.path.join("results", "random_forest_report.txt"), "w") as f:
    f.write("Random Forest Results:\n")
    f.write(report)

cm = confusion_matrix(y_test, rf_predictions)
plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    xticklabels=["Legitimate", "Fraud"],
    yticklabels=["Legitimate", "Fraud"],
)
plt.title("Confusion Matrix - Random Forest")
plt.ylabel("Actual Class")
plt.xlabel("Predicted Class")
plt.savefig(os.path.join("results", "random_forest_confusion_matrix.png"))


# Building/Trainig Simple DNN Model

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(
    y_train.values if hasattr(y_train, "values") else y_train
).unsqueeze(1)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(
    y_test.values if hasattr(y_test, "values") else y_test
).unsqueeze(1)

val_size = int(0.2 * len(X_train_tensor))
train_size = len(X_train_tensor) - val_size
train_dataset, val_dataset = torch.utils.data.random_split(
    TensorDataset(X_train_tensor, y_train_tensor), [train_size, val_size]
)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)


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


input_dim = X_train_scaled.shape[1]
model = DNNClassifier(input_dim)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


print("Training DNN")

best_val_loss = float("inf")
patience, patience_counter = 4, 0
best_weights = None

for epoch in range(50):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()

    model.eval()
    val_losses = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            val_losses.append(criterion(model(X_batch), y_batch).item())
    val_loss = np.mean(val_losses)

    print(f"Epoch {epoch + 1:2d}/20 — val_loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

model.load_state_dict(best_weights)

print("\n--- Testing the DNN on Unseen Data ---")
model.eval()
with torch.no_grad():
    dnn_probabilities = model(X_test_tensor).numpy()

dnn_predictions = (dnn_probabilities > 0.5).astype(int)

report = classification_report(y_test, dnn_predictions)
print("Deep Neural Network Results:")
print(report)
with open("results\\dnn_report.txt", "w") as f:
    f.write("Deep Neural Network Results:\n")
    f.write(report)

torch.save(
    {
        "model_state_dict": best_weights,
        "model_config": {"input_dim": input_dim},
        "scaler": scaler,
    },
    "results\\dnn_model.pt",
)
print("Model saved to dnn_model.pt")


cm = confusion_matrix(y_test, dnn_predictions)
plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    xticklabels=["Legitimate", "Fraud"],
    yticklabels=["Legitimate", "Fraud"],
)
plt.title("Confusion Matrix - Deep Neural Network")
plt.ylabel("Actual Class")
plt.xlabel("Predicted Class")
plt.savefig(os.path.join("results", "dnn_confusion_matrix.png"))

# Load DNN Model
checkpoint = torch.load(os.path.join("results", "dnn_model.pt"), weights_only=False)

loaded_model = DNNClassifier(checkpoint["model_config"]["input_dim"])
loaded_model.load_state_dict(checkpoint["model_state_dict"])
loaded_model.eval()

scaler = checkpoint["scaler"]
print("Model loaded successfully")



# REAL-WORLD SIMULATION
print("\n--- Creating Real-World Imbalanced Test Set (99:1) ---")
test_df = X_test.copy()
test_df["Class"] = y_test

legit_test = test_df[test_df["Class"] == 0]
fraud_test = test_df[test_df["Class"] == 1]

# only 1% of fraud
num_fraud_to_keep = int(len(legit_test) * 0.01)
fraud_test_sampled = fraud_test.sample(n=num_fraud_to_keep, random_state=42)

real_world_test = pd.concat([legit_test, fraud_test_sampled])
X_real_world = real_world_test.drop("Class", axis=1)
y_real_world = real_world_test["Class"]

print(f"Real-World Test Set Shape: {X_real_world.shape}")
print(f"Legitimate (0): {len(legit_test)} | Fraud (1): {len(fraud_test_sampled)}")

# Process Real-World Data for PyTorch
X_rw_scaled = scaler.transform(X_real_world)
X_rw_tensor = torch.FloatTensor(X_rw_scaled)

with torch.no_grad():
    dnn_rw_probabilities = loaded_model(X_rw_tensor).numpy().flatten()

dnn_rw_predictions_default = (dnn_rw_probabilities > 0.5).astype(int)

report_rw = classification_report(y_real_world, dnn_rw_predictions_default)
print("\nDeep Neural Network Real-World Results (Threshold 0.5 fixed):")
print(report_rw)
with open("results\\dnn_real_world_fixed_threshold_report.txt", "w") as f:
    f.write("DNN Real-World Results (Threshold 0.5 fixed):\n")
    f.write(report_rw)

cm_fixed = confusion_matrix(y_real_world, dnn_rw_predictions_default)
plt.figure(figsize=(6, 4))
sns.heatmap(
    cm_fixed,
    annot=True,
    fmt="d",
    cmap="Purples",
    cbar=False,
    xticklabels=["Legitimate", "Fraud"],
    yticklabels=["Legitimate", "Fraud"],
)
plt.title(f"DNN Confusion Matrix (Threshold 0.5 fixed)")
plt.ylabel("Actual Class")
plt.xlabel("Predicted Class")
plt.savefig(os.path.join("results", "dnn_real_world_fixed_threshold_confusion_matrix.png"))


# ALGORITHMIC THRESHOLD OPTIMIZATION
print("\n--- Algorithmic Threshold Optimization ---")
precisions, recalls, thresholds = precision_recall_curve(
    y_real_world, dnn_rw_probabilities
)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)

# Find optimal threshold
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]
optimal_f1 = f1_scores[optimal_idx]

print(f"Algorithm Complete!")
print(f"Mathematically optimal threshold: {optimal_threshold:.4f}")
print(f"Expected F1-Score: {optimal_f1:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(
    thresholds, f1_scores[:-1], label="F1-Score Curve", color="purple", linewidth=2
)
plt.axvline(
    x=optimal_threshold,
    color="red",
    linestyle="--",
    label=f"Optimal Threshold ({optimal_threshold:.4f})",
)
plt.title("Algorithmic Search for Optimal Decision Threshold")
plt.xlabel("Probability Threshold")
plt.ylabel("F1-Score")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join("results", "threshold_optimization_curve.png"))

# Final Test with Optimal Threshold
dnn_rw_predictions_optimal = (dnn_rw_probabilities > optimal_threshold).astype(int)

report_rw = classification_report(y_real_world, dnn_rw_predictions_optimal)
print(f"\nDeep Neural Network Real-World Results (With optimal threshold {optimal_threshold:.4f}):")
print(report_rw)
with open(os.path.join("results", "dnn_real_world_optimal_threshold_report.txt"), "w") as f:
    f.write(f"DNN Real-World Results With Optimal Threshold (Threshold {optimal_threshold:.4f}):\n")
    f.write(report_rw)

cm_optimal = confusion_matrix(y_real_world, dnn_rw_predictions_optimal)
plt.figure(figsize=(6, 4))
sns.heatmap(
    cm_optimal,
    annot=True,
    fmt="d",
    cmap="Purples",
    cbar=False,
    xticklabels=["Legitimate", "Fraud"],
    yticklabels=["Legitimate", "Fraud"],
)
plt.title(f"DNN Confusion Matrix (Optimal Threshold: {optimal_threshold:.4f})")
plt.ylabel("Actual Class")
plt.xlabel("Predicted Class")
plt.savefig(os.path.join("results", "dnn_real_world_optimal_threshold_confusion_matrix.png"))

print("\nAll Complete")
