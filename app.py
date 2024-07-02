import os
from PIL import Image
import tensorflow as tf
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import numpy as np



app = Flask(__name__)


class_dict={'Tomato Bacterial spot': 0,
            'Tomato Early blight': 1,
            'Tomato Late blight': 2,
            'Tomato Leaf Mold': 3,
            'Tomato Septoria leaf spot': 4,
            'Tomato Spider mites Two-spotted spider mite': 5,
            'Tomato Target Spot': 6,
            'Tomato Tomato Yellow Leaf Curl Virus': 7,
            'Tomato Tomato mosaic virus': 8,
            'Tomato healthy': 9}

def prepare(image):
    img_array=image/255
    return img_array.reshape(-1,128,128,3)

def load_image(image_path):
    img=Image.open(image_path)
    img=img.resize((128,128))
    return img

def prediction_cls(prediction):
    for key, clss in class_dict.items():
        if np.argmax(prediction)==clss:
            return key

@app.route('/test')
def index():
    return 'Hello, World!'

app.config['UPLOAD_FOLDER'] = 'static/uploads/'


@app.route('/', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        # Save the uploaded image
        filename = secure_filename(image.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(image_path)
        img=load_image(image_path)
 
        img=tf.keras.preprocessing.image.img_to_array(img)
        model=tf.keras.models.load_model("model_vgg19.h5")
        #print(img)
        img=prepare(img)
        #print(img)
        prediction=prediction_cls(model.predict(img))
        # return render_template('submit.html',image_path=image_path,
        #                        modified_image_path=new_image_path)
        return render_template('submit.html',image_path=image_path,prediction=prediction,)
    return render_template('submit.html')

if __name__ == '__main__':
    app.run()