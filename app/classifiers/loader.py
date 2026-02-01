from transformers import AutoTokenizer, AutoModelForSequenceClassification

class ModelBundle:
    def __init__(self, path):
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path)

# Load models for each rubric category
current_state = ModelBundle(r"app\models\bert-base-uncased\current_state_lr2e-05_bs32_ep3")
next_steps = ModelBundle(r"app\models\google_electra-base-discriminator\next_steps_lr1e-05_bs32_ep5")
strategy = ModelBundle(r"app\models\google_electra-base-discriminator\strategy_lr3e-05_bs16_ep3")