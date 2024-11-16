from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Load the fine-tuned GPT-2 model and tokenizer
model_name = "longformer-base-4096"

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained(model_name)

from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

# Define the dataset for training
train_file = "hamlet_play.txt"
test_file = "dialogue/hamlet.txt"


def load_dataset(train_file, test_file, tokenizer):
    train_dataset = TextDataset(tokenizer=tokenizer, file_path=train_file, block_size=128)
    test_dataset = TextDataset(tokenizer=tokenizer, file_path=test_file, block_size=128)
    return train_dataset, test_dataset


train_dataset, test_dataset = load_dataset(train_file, test_file, tokenizer)

# Define the data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Set up the training arguments
training_args = TrainingArguments(
    output_dir="output",
    overwrite_output_dir=True,
    num_train_epochs=50,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    save_steps=10_000,
    save_total_limit=2,
    evaluation_strategy="epoch",
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train the model
trainer.train()

# Save the fine-tuned model
# trainer.save_model("fine_tuned_gpt2")

trainer.save_model("fine_tuned_longformer-Macbeth")
tokenizer.save_pretrained("fine_tuned_longformer-Macbeth")
