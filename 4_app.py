from flask import Flask, render_template
import joblib

app = Flask(__name__)
model = joblib.load('model.joblib')
klasy = ['setosa', 'versicolor', 'virginica']

# while True:
#     sepal_length = float(input("Sepal lenght: "))
#     sepal_width =  float(input("Sepal width:  "))
#     petal_length = float(input("Petal lenght: "))
#     petal_width =  float(input("Petal width:  "))
#
#     cechy = [[sepal_length, sepal_width, petal_length, petal_width]]
#     y_pred = model.predict(cechy)[0]
#
#     #print(f"Wynik: {data.target_names[y_pred]}")
#     print(f"Wynik: {klasy[y_pred]}")

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)