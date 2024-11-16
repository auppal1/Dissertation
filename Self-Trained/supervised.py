import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

question = []
answer = []
with open('tybalt_qa.txt', 'r', encoding='cp1252', errors='ignore') as f:
    for line in f:
        line = line.strip()  # Remove leading and trailing whitespace
        if '|' in line:  # Ensure there is a delimiter
            parts = line.split('|')
            if len(parts) >= 2:  # Check that both question and answer are present
                question.append(parts[0].strip())  # Strip whitespace from question
                answer.append(parts[1].strip())  # Strip whitespace from answer
            else:
                print("Skipping incomplete line:", line)
        else:
            print("Skipping line without delimiter:", line)

print(len(question) == len(answer))  # This should now be True

result = pd.DataFrame({"question": question, "answer": answer})

# Load the dataset
df = result.head(100)

# Create a pipeline that consists of a CountVectorizer, a TfidfTransformer, and a RandomForestClassifier
pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', RandomForestClassifier())
])

# Fit the pipeline to the text data
pipeline.fit(df['question'], df['answer'])

# Test the model
try:
    while True:
        user_input = input('You: ')
        if user_input.lower() == 'quit':
            break
        response = pipeline.predict([user_input])[0]
        print('Bot:', response)
except KeyboardInterrupt:
    print("Interrupted by user")
