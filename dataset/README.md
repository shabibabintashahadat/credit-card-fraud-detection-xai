# Dataset Setup

This project uses the **Credit Card Fraud Detection Dataset (2023)**.

### Download Instructions:
1. Download `creditcard_2023.csv` from Kaggle:
   - [Kaggle Dataset: Credit Card Fraud Detection 2023](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023)
2. Place the file inside this directory:
   ```text
   dataset/creditcard_2023.csv
   ```

### Dataset Structure:
- **Total records:** 568,630 transactions
- **Features:** 
  - `id`: Unique identifier (dropped during preprocessing)
  - `V1`–`V28`: Anonymized PCA components protecting cardholder privacy
  - `Amount`: Monetary transaction amount (normalized using `RobustScaler`)
  - `Class`: Binary ground truth target (`0`: Legitimate, `1`: Fraudulent)
