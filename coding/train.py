import pandas as pd

import warnings
warnings.filterwarnings('ignore')

from transformers import (AutoModel,AutoModelForMaskedLM,
                          AutoTokenizer, LineByLineTextDataset,
                          DataCollatorForLanguageModeling,
                          Trainer, TrainingArguments)


train_data = pd.read_csv('./input/train.csv')
test_data = pd.read_csv('./input/test.csv')

data = pd.concat([train_data,test_data])
data['excerpt'] = data['excerpt'].apply(lambda x: x.replace('\n',''))

text = '\n'.join(data.excerpt.tolist())

with open('text.txt', 'w', encoding='utf-8') as f:
    f.write(text)

model_name = 'roberta-base'
model = AutoModelForMaskedLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained('./clrp_roberta_base')


train_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="text.txt", #mention train text file here
    block_size=256)

valid_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="text.txt", #mention valid text file here
    block_size=256)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

training_args = TrainingArguments(
    output_dir="./clrp_roberta_base_chk", #select model path for checkpoint
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy= 'steps',
    save_total_limit=2,
    eval_steps=500,
    metric_for_best_model='eval_loss',
    greater_is_better=False,
    load_best_model_at_end =True,
    prediction_loss_only=True,
    report_to = "none")

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset)


trainer.train()
trainer.save_model(f'./clrp_roberta_base')
