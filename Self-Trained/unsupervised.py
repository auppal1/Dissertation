import pandas as pd

question = []
answer = []
with open("tybalt_qa.txt", 'r', encoding='cp1252', errors='ignore') as f:
    for line in f:
        line = line.split('|')

        question.append(line[0])
        answer.append(line[1])
print(len(question) == len(answer))

result = pd.DataFrame({"question": question, "answer": answer})
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# load the dataset
df = result

# initialize lemmatizer
lemmatizer = WordNetLemmatizer()


# tokenize and lemmatize the text
def tokenize_and_lemmatize(text):
    tokens = nltk.word_tokenize(text.lower())
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens


# create a TfidfVectorizer object
vectorizer = TfidfVectorizer(tokenizer=tokenize_and_lemmatize)

# fit the vectorizer to the text data
vectorizer.fit(df['question'])

# transform the text data to a tf-idf matrix
tfidf_matrix = vectorizer.transform(df['question'])

# create a dictionary with question-answer pairs
qa_dict = dict(zip(df['question'], df['answer']))


# define a function to generate responses
def generate_response(user_input):
    # transform the user input to a tf-idf vector
    user_tfidf = vectorizer.transform([user_input])
    # calculate the cosine similarities between the user input and the questions in the dataset
    similarities = cosine_similarity(user_tfidf, tfidf_matrix)
    # get the index of the question with the highest similarity
    index = similarities.argmax()
    # get the corresponding answer from the qa_dict
    response = qa_dict[df.iloc[index]['question']]
    return response


# test the chatbot
while True:
    try:
        user_input = input('You: ')
        if user_input.lower() == 'quit':
            break
        response = generate_response(user_input)
        print('Bot:', response)

    except KeyboardInterrupt:
        print("Interrupted by user")
        break
