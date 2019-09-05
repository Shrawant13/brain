from django.shortcuts import render
from keras.preprocessing import image
import numpy as np
import tensorflow as tf
from keras.models import load_model
global graph,model
from PIL import Image

#initializing the graph
graph = tf.get_default_graph()

#loading our trained model
print("Keras model loading.......")
model = load_model('Tumor recognition app/AlexNetModel.hdf5')
print("Model loaded!!")

#creating a dictionary of classes
class_dict = {'No Tumor Detected': 0,
            'Tumor Detected': 1}

class_names = list(class_dict.keys())

def prediction(request):
    if request.method == 'POST' and request.FILES['myfile']:
        post = request.method == 'POST'
        myfile = request.FILES['myfile']
        img = image.load_img(myfile,target_size=(50, 50))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img/50
        with graph.as_default():
            preds = model.predict(img)
        preds = preds.flatten()
        m = max(preds)
        for index, item in enumerate(preds):
            if item == m:
                result = class_names[index]
        return render(request, "Tumor recognition app/prediction.html", {
            'result': result})
    else:
        return render(request, "Tumor recognition app/prediction.html")
