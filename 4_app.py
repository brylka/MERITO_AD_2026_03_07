from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load('model.joblib')
klasy = ['setosa', 'versicolor', 'virginica']

@app.route("/", methods=['GET', 'POST'])
def index():
    wynik = None
    if request.method == 'POST':
        cechy = [[float(request.form['sepal_length']),
                  float(request.form['sepal_width']),
                  float(request.form['petal_length']),
                  float(request.form['petal_width'])]]
        wynik = klasy[model.predict(cechy)[0]]
    return render_template("index.html", wynik=wynik)

if __name__ == "__main__":
    app.run(debug=True)