from flask import Flask, request, jsonify
from flask import render_template
import pandas as pd
import numpy as np
import pickle

user_final_rating = pd.read_pickle("models/user_final_rating.pkl")
df = pd.read_csv("models/df_old.csv")
word_vectorizer = pickle.load(
    open('models/word_vectorizer.pkl', 'rb'))
logit = pickle.load(
    open('models/logit_model.pkl', 'rb'))


def recommendations(user_input):
    # Add the unuser check for now as we do not have reviewes for him
    # The recommendations would be random for that person.
    try:
        print(user_final_rating.loc[user_input])
    except:
        return ''

    if user_final_rating.loc[user_input] is not None:
        d = user_final_rating.loc[user_input].sort_values(ascending=False)[0:20]
        i = 0
        a = {}
        for prod_name in d.index.tolist():
            product_name = prod_name
            print(product_name)
            product_name_review_list = df[df['product_name'] == product_name]['Review'].tolist()
            if len(product_name_review_list) > 0:
                print("Should go into vectorized")
                features = word_vectorizer.transform(product_name_review_list)
                logit.predict(features)
                a[product_name] = logit.predict(features).mean() * 100
        print(a)
        b = pd.Series(a).sort_values(ascending=False).head(5).index.tolist()
        print(b)
        return b


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    username = str(request.form.get('username'))
    recommendations1 = recommendations(username)
    print("Output :", recommendations1)
    return render_template('index.html', username=username,
                           recommendations_user='Your Top 5 Recommendations are:\n {}'.format(recommendations1))


if __name__ == "__main__":
    app.run()

