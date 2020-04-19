import sys
from flask import Flask, request, jsonify, redirect, render_template, url_for
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
import tensorflow as tf
from flask_session import Session
#import dataset_creator
import numpy as np
#from keras.models import load_model, model_from_json
import json
import cv2
import os

app = Flask(__name__)
print(tf.__version__,"----------------------")
if not os.getenv("DATABASE_URL"):
    raise RuntimeError("DATABASE_URL is not set")
# Settings
#app.config['DEBUG'] = True
app.config["IMAGE_UPLOADS"] = "./static/images"
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgres://tvviudmrjxggzt:0e1dcfbbf5d37a6031e5f0d4fda56d42289c591c67a7aa4d459eb0b36ab9c5fc@ec2-54-22205-79.compute-1.amazonaws.com:5432/dats98s4nmt92g'
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)
engine = create_engine(os.getenv("DATABASE_URL"))
db = scoped_session(sessionmaker(bind=engine))


DIAGNOSTICS = {
    0: 'covid-19',
    1: 'Pneumonia',
    2: 'Normal image'
}
PRUEBA = False

if not PRUEBA:
    with open('./architecture/model_config_new.json') as json_file:
        json_config = json_file.read()
    model = tf.keras.models.model_from_json(json_config)
    try:
        model.load_weights('./checkpoints/covidresnet50Weights_new.h5')
        print("weights were successfully loaded onto model.")
    except:
        raise Exception("weights could no be loaded")


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/listado', methods=['GET', 'POST'])
def list_db():
    return render_template('listado.html')


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save(os.path.join(app.config['IMAGE_UPLOADS'], f.filename))
        image = cv2.imread(os.path.join(
            app.config['IMAGE_UPLOADS'], f.filename))
        image = cv2.resize(image, (224, 224))
        image = np.reshape(image, [-1, 224, 224, 3]).astype(np.float32)
        if PRUEBA:
            diagnostico = DIAGNOSTICS.get(0)
            prediction = [0.9, 0.5, 0.12]
        else:
            prediction = model(image)
            diagnostico = DIAGNOSTICS.get(np.argmax(prediction))
            prediction = tf.nn.softmax(prediction).numpy()[0]
        return render_template("resultado.html",
                               fname=f"{request.form.get('primNombre')} {request.form.get('segNombre')}",
                               lname=f"{request.form.get('apPaterno')} {request.form.get('apMaterno')}",
                               cedula=request.form.get('cedula'),
                               imagen=os.path.join("images", f.filename),
                               diagnostico=diagnostico,
                               pred=prediction
                               )

    return redirect(url_for('index'))
