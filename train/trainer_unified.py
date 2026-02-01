"""
Unified rubric-based fine-tuning 
"""

import os, re, random, argparse, yaml, json, pandas as pd, numpy as np, torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, cohen_kappa_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from torch.nn import CrossEntropyLoss
import matplotlib.pyplot as plt
os.environ['TORCH_LOAD_SAFE'] = '0'

# -------------------- Parse arguments --------------------
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
args = parser.parse_args()

# -------------------- Defaults --------------------
rubrics = ["current_state", "strategy", "next_steps"]
param_grid = {
    "learning_rate": [1e-5, 2e-5, 3e-5],
    "batch_size": [8, 16, 32],
    "epochs": [3, 5]
}
models_to_run = ["google/electra-base-discriminator"]  # can include multiple models
LABEL_NAME = "binary_label"
SEED = 42
expid = "default"

# -------------------- Load config if provided --------------------
if args.config:
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    rubrics = cfg.get("rubrics", rubrics)
    param_grid = cfg.get("param_grid", param_grid)
    models_to_run = cfg.get("models", models_to_run)
    LABEL_NAME = cfg.get("label_name", LABEL_NAME)
    expid = cfg.get("expid", "exp001")
    print(f" Loaded config {args.config}")
else:
    print(" Using default settings (no config file)")

# -------------------- Setup --------------------
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

base_output_root = "outputs_new"

# -------------------- Utility functions --------------------
def normalize_text(s):
    s = str(s).lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "balanced_accuracy": balanced_accuracy_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "f1": f1_score(labels, preds),
        "kappa": cohen_kappa_score(labels, preds),
    }

class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = CrossEntropyLoss(weight=self.class_weights.to(model.device))
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# -------------------- Training Loop --------------------
for model_name in models_to_run:
    print(f"\n==============================")
    print(f" Starting model: {model_name}")
    print(f"==============================")

    base_output = os.path.join(base_output_root, model_name.replace("/", "_"))
    os.makedirs(f"{base_output}/models", exist_ok=True)
    os.makedirs(f"{base_output}/logs", exist_ok=True)
    os.makedirs(f"{base_output}/images", exist_ok=True)

    results_csv = f"{base_output}/logs/all_results_{expid}.csv"
    all_results = []

    for rubric in rubrics:
        print(f"\n Rubric: {rubric}")
        dataset_path = f"data/processed/preprocessed_dataset_{rubric}.csv"
        if not os.path.exists(dataset_path):
            print(f" Missing dataset for {rubric}, skipping...")
            continue

        df = pd.read_csv(dataset_path)
        df["norm_text"] = df["Selected Text"].apply(normalize_text)
        df = df.drop_duplicates(subset=["norm_text", LABEL_NAME])

        # Use AICC+ASE for train/val, ADA+ICC2 for test
        train_val_df = df[~df['Course'].isin(['ADA', 'ICC2'])]
        test_df = df[df['Course'].isin(['ADA', 'ICC2'])]
        
        # Further split train_val into train (85%) and val (15%)
        train_df, val_df = train_test_split(
            train_val_df, test_size=0.15, stratify=train_val_df[LABEL_NAME], random_state=SEED
        )

        print(f"  Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
        print(f"  Val:   {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
        print(f"  Test:  {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        def preprocess(batch):
            enc = tokenizer(batch["Selected Text"], truncation=True, padding=True)
            enc["labels"] = batch[LABEL_NAME]
            return enc

        train_df.to_csv(f"tmp_train_{rubric}.csv", index=False)
        val_df.to_csv(f"tmp_val_{rubric}.csv", index=False)
        test_df.to_csv(f"tmp_test_{rubric}.csv", index=False)  

        dataset = {
            "train": load_dataset("csv", data_files=f"tmp_train_{rubric}.csv")["train"],
            "val": load_dataset("csv", data_files=f"tmp_val_{rubric}.csv")["train"],
            "test": load_dataset("csv", data_files=f"tmp_test_{rubric}.csv")["train"], 
        }

        train_ds = dataset["train"].map(preprocess, batched=True)
        val_ds = dataset["val"].map(preprocess, batched=True)
        test_ds = dataset["test"].map(preprocess, batched=True) 

        pos_weight = len(train_df[train_df[LABEL_NAME]==0]) / max(len(train_df[train_df[LABEL_NAME]==1]), 1)
        class_weights = torch.tensor([1.0, pos_weight])

        for lr in param_grid["learning_rate"]:
            for bs in param_grid["batch_size"]:
                for ep in param_grid["epochs"]:
                    print(f" {model_name} | {rubric} | lr={lr} | bs={bs} | ep={ep}")

                    output_dir = f"{base_output}/models/{rubric}_lr{lr}_bs{bs}_ep{ep}"
                    os.makedirs(output_dir, exist_ok=True)

                    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

                    args = TrainingArguments(
                        output_dir=output_dir,
                        eval_strategy="epoch",
                        save_strategy="no",
                        learning_rate=lr,
                        per_device_train_batch_size=bs,
                        per_device_eval_batch_size=bs,
                        num_train_epochs=ep,
                        weight_decay=0.01,
                        fp16=True,
                        logging_dir=f"{output_dir}/logs",
                        logging_steps=50,
                        report_to="none",
                        seed=SEED,
                    )

                    trainer = WeightedTrainer(
                        model=model,
                        args=args,
                        train_dataset=train_ds,
                        eval_dataset=val_ds,
                        tokenizer=tokenizer,
                        compute_metrics=compute_metrics,
                        class_weights=class_weights,
                    )

                    trainer.train()
                    
                    test_predictions = trainer.predict(test_ds)
                    test_labels = test_predictions.label_ids
                    test_preds = np.argmax(test_predictions.predictions, axis=1)

                    # --- Metrics (using TEST set) ---
                    metrics = {
                        "model": model_name,
                        "rubric": rubric,
                        "lr": lr,
                        "batch_size": bs,
                        "epochs": ep,
                        "accuracy": accuracy_score(test_labels, test_preds),
                        "balanced_accuracy": balanced_accuracy_score(test_labels, test_preds),
                        "precision": precision_score(test_labels, test_preds),
                        "recall": recall_score(test_labels, test_preds),
                        "f1": f1_score(test_labels, test_preds),
                        "kappa": cohen_kappa_score(test_labels, test_preds),
                    }

                    # --- Save model ---
                    model.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    #  Confusion Matrix + Predictions 
                    cm = confusion_matrix(test_labels, test_preds)  
                    cm_df = pd.DataFrame(cm, index=["Class 0", "Class 1"], columns=["Pred 0", "Pred 1"])
                    cm_csv_path = f"{base_output}/logs/{rubric}_lr{lr}_bs{bs}_ep{ep}_cm.csv"
                    cm_png_path = f"{base_output}/images/{rubric}_lr{lr}_bs{bs}_ep{ep}_cm.png"
                    preds_path = f"{base_output}/logs/{rubric}_lr{lr}_bs{bs}_ep{ep}_predictions.csv"

                    cm_df.to_csv(cm_csv_path)
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Class 0", "Class 1"])
                    disp.plot(cmap="Blues", xticks_rotation=45)
                    plt.title(f"{rubric} | lr={lr} | bs={bs} | ep={ep}")
                    plt.tight_layout()
                    plt.savefig(cm_png_path)
                    plt.close()

                   
                    preds_df = pd.DataFrame({
                        "text": test_df["Selected Text"].values,  #
                        "true_label": test_labels,              
                        "pred_label": test_preds,                
                    })
                    preds_df.to_csv(preds_path, index=False)

                    all_results.append(metrics)
                    pd.DataFrame(all_results).to_csv(results_csv, index=False)

    print(f" Finished model: {model_name}")
    print(f" Results: {results_csv}")

print("\n All models + rubrics complete.")