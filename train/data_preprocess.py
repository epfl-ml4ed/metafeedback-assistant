import pandas as pd
import os

# Load raw data
df = pd.read_excel("data/raw/translated_sentence_annotations.xlsx")

# Source column contains: "AICC", "ADA", "ASE", "ICC1", "ICC2", "Analysis"
df['Course'] = df['Source'].str.extract(r'(AICC|ADA|ASE|ICC1|ICC2|Analysis)', expand=False)

# Drop unnecessary columns
df.drop(["id", "start", "end"], axis=1, inplace=True)

# Combine "Specific Strategy" (4) and "General Learning Strategy" (5)
# under one unified label called "Strategy".
label_map = {
    2: "Current State",
    3: "Next Steps",
    4: "Strategy",  # "Specific Strategy"
    5: "Strategy"   #  "General Learning Strategy"
}

# Convert numeric label IDs to readable text; others become "Other"
df["labelID"] = df["labelID"].apply(lambda x: label_map.get(x, "Other"))

# remove empty rows or duplicates
df = df.dropna(subset=["Selected Text", "labelID"])
df = df.drop_duplicates(subset=["Selected Text", "labelID", "Source"], keep="first")
df = df.sort_values(by=["Filename", "Index"], ascending=True).reset_index(drop=True)

#  Create binary datasets for each rubric 
target_labels = ["Current State", "Next Steps", "Strategy"]
os.makedirs("data/processed", exist_ok=True)

for lbl in target_labels:
    df_lbl = df.copy()
    df_lbl["binary_label"] = (df_lbl["labelID"] == lbl).astype(int)
    
    # Include Course column in output for later train/val/test split
    df_lbl[["Selected Text", "binary_label", "Course"]].to_csv(
        f"data/processed/preprocessed_dataset_{lbl.replace(' ', '_').lower()}.csv",
        index=False
    )
    