import pandas as pd
import json
from sklearn.preprocessing import StandardScaler

def clinical_process():
    train_raw               = pd.read_csv("Raw train clinical data file")
    val_raw                 = pd.read_csv("Raw Val clinical data file")
    test_raw                = pd.read_csv("Raw test clinical data file")
    clinical_numerical_cols = ["Age", "BMI", "FEV1 Predicted", "DLCO Predicted", "Pack-Years Of Cigarette Use"]
    
    scaler                             = StandardScaler()
    scaler.fit(train_raw[clinical_numerical_cols])
    train_raw[clinical_numerical_cols] = scaler.transform(train_raw[clinical_numerical_cols])
    val_raw[clinical_numerical_cols]   = scaler.transform(val_raw[clinical_numerical_cols])
    test_raw[clinical_numerical_cols]  = scaler.transform(test_raw[clinical_numerical_cols])

    if train_raw.isnull().any().any():
        print("There is at least one null value in the Training Dataframe.")
    else:
        print("No null values in the DataFrame.")
    if val_raw.isnull().any().any():
        print("There is at least one null value in the Validation Dataframe.")
    else:
        print("No null values in the DataFrame.")
    if test_raw.isnull().any().any():
        print("There is at least one null value in the Testing Dataframe.")
    else:
        print("No null values in the DataFrame.")
    
    for col in train_raw.select_dtypes(include="object").columns:
        train_raw[col] = train_raw[col].astype(str).str.strip()
    for col in val_raw.select_dtypes(include="object").columns:
        val_raw[col] = val_raw[col].astype(str).str.strip()
    for col in test_raw.select_dtypes(include="object").columns:
        test_raw[col] = test_raw[col].astype(str).str.strip()

    train_raw.to_csv("Processed train clinical data file", index = False)
    val_raw.to_csv("Processed Val clinical data file", index = False)
    test_raw.to_csv("Processed test clinical data file", index = False)

if __name__ == "__main__":
    clinical_process()