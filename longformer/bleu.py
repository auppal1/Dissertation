from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import sacrebleu

# Load the fine-tuned model and tokenizer
model_name = "fine_tuned_longformer-Macbeth"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()


# Function to generate predictions from the model
def generate_text(input_text, model, tokenizer, max_length=50):
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=max_length)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


# Calculate the BLEU score
def calculate_bleu_score(candidate_corpus, reference_corpus):
    bleu_score = sacrebleu.corpus_bleu(candidate_corpus, [reference_corpus])
    return bleu_score.score


# Example source and reference data
source_texts = "hamlet_play.txt"
reference_texts = "hamletdaatd.txt"

# Generate translations using the model
generated_texts = [generate_text(text, model, tokenizer) for text in source_texts]

# Calculate BLEU score
bleu_score = calculate_bleu_score(generated_texts, reference_texts)
print(f"BLEU Score: {bleu_score}")
