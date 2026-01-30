from transformers import AutoTokenizer, AutoModelForSequenceClassification

class ModelBundle:
    def __init__(self, path):
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path)

# Load your three models
current_state = ModelBundle(r"C:\repos\semester_project\metafeedback_feedback_on_feedback\outputs\google_electra-base-discriminator\models\current_state_lr2e-05_bs8_ep3")
next_steps = ModelBundle(r"C:\repos\semester_project\metafeedback_feedback_on_feedback\outputs\google_electra-base-discriminator\models\next_steps_lr1e-05_bs32_ep5")
strategy = ModelBundle(r"C:\repos\semester_project\metafeedback_feedback_on_feedback\outputs\google_electra-base-discriminator\models\strategy_lr3e-05_bs16_ep3")