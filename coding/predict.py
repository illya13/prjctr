import numpy as np
from transformers import (AutoModel,AutoModelForMaskedLM,
                          AutoTokenizer, LineByLineTextDataset,
                          DataCollatorForLanguageModeling,
                          Trainer, TrainingArguments)


model = AutoModelForMaskedLM.from_pretrained(f'./clrp_roberta_base')
tokenizer = AutoTokenizer.from_pretrained(f'./clrp_roberta_base')

trainer = Trainer(model=model, tokenizer=tokenizer)

train_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="test.txt",
    block_size=256)

prediction_output = trainer.predict(train_dataset)
print(prediction_output)

predictions = prediction_output.predictions
final_prediction = np.mean(predictions, axis=0)
print(final_prediction)