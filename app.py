from flask import Flask, render_template, request
from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
# from keras.applications import imagenet_utils
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import tensorflow as tf


from PIL import Image

from werkzeug.utils import secure_filename

import numpy as np
import flask
import io
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/img/inputImages'
image_path_global = None


model = None

def load_model():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    global model
    model = ResNet50(weights="imagenet")
    global graph
    graph = tf.get_default_graph()

def prepare_image(image_path, target):
    img = image.load_img(image_path, target_size=target)
    img = image.img_to_array(img)

    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    return img

@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}
    predictions =[]

    name = request.form.get('name')
    iname = request.form['image_name']

    if len(iname) ==0:
        iname = "None"

    request.form.get('email')
    imagefile = request.files['imagefile']

    filename = secure_filename(imagefile.filename)
    imagefile.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image_path_global = "../"+image_path
    model_name = request.form['classificationModel']

    img = prepare_image(image_path, target=(224, 224))

    with graph.as_default():
        preds = model.predict(img)
    results = decode_predictions(preds)
    
    data["predictions"] = []

    for (imagenetID, label, prob) in results[0][:2]:
        r = {"label": label, "probability": float(prob)}
        data["predictions"].append(r)
        prob2 = "{0:.3f}".format(prob)
        res = label +","+str(prob2)+"\n"
        predictions.append(res)

    data["success"] = True
    # print(predictions)

    # os.remove(image_path)
    # return flask.jsonify(data)
    print(image_path_global)
    return render_template('nimbus_image_output.html',
                model_name = model_name,
                iname = iname,
                image_path=image_path_global,
                predictions=predictions)

@app.route('/',methods=['GET', 'POST'])
def nimbus_image():
    for the_file in os.listdir(app.config['UPLOAD_FOLDER']):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], the_file)
        os.remove(file_path)
    
    return render_template('nimbus_image.html')

if __name__ == '__main__':
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    load_model()
    
    app.run(debug=True)#,threaded = False)