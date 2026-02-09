from flask import Flask, render_template, request, redirect, url_for, session, abort
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
import os

app = Flask(__name__)
app.secret_key = "student-performance-secret"

@app.route("/", methods=["GET"])
def home():
    result = session.pop("result", None)
    return render_template("home.html", results=result)

@app.route("/predictdata", methods=["POST"])
def predict_datapoint():
    data = CustomData(
        gender=request.form["gender"],
        race_ethnicity=request.form["ethnicity"],
        parental_level_of_education=request.form["parental_level_of_education"],
        lunch=request.form["lunch"],
        test_preparation_course=request.form["test_preparation_course"],
        reading_score=float(request.form["reading_score"]),
        writing_score=float(request.form["writing_score"]),
    )

    pred_df = data.get_data_as_data_frame()
    pipeline = PredictPipeline()
    result = round(float(pipeline.predict(pred_df)[0]), 2)

    session["result"] = result
    return redirect(url_for("home"))

# 🚨 HARD BLOCK: any GET to /predictdata
@app.route("/predictdata", methods=["GET"])
def block_get_predict():
    abort(405)   # Method Not Allowed

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
