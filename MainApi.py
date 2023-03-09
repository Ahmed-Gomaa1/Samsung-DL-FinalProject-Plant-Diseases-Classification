from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

app = Flask(__name__)

dic = {0:'Apple_scab', 1:'Apple_Black_rot', 2:'Cedar_apple_rust', 
         3:'Apple_healthy',4: 'Blueberry_healthy', 
        5: 'Cherry_Powdery_mildew', 6:'Cherry_healthy', 
         7:'Corn_Cercospora_leaf_spot', 8:'Corn_Common_rust_', 
         9:'Corn_Northern_Leaf_Blight',10: 'Corn_healthy', 
         11:'Grape_Black_rot', 12:'Grape_Black_Measles', 
         13:'Grape_Leaf_blight', 14:'Grape_healthy', 
         15:'Orange_Haunglongbing', 16:'Peach_Bacterial_spot', 
         17:'Peach_healthy',18: 'Pepper,_bell_Bacterial_spot', 19:'Pepper,_bell_healthy', 
         20:'Potato_Early_blight', 21:'Potato_Late_blight',22: 'Potato_healthy', 
         23:'Raspberry_healthy',24: 'Soybean_healthy',25: 'Squash_Powdery_mildew', 
         26:'Strawberry_Leaf_scorch', 27:'Strawberry_healthy', 28:'Tomato_Bacterial_spot', 
         29:'Tomato_Early_blight', 30:'Tomato_Late_blight',31: 'Tomato_Leaf_Mold', 
         32:'Tomato_Septoria_leaf_spot', 33:'Tomato_Spider_mites Two-spotted_spider_mite', 
         34:'Tomato_Target_Spot', 35:'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 
        36: 'Tomato_Tomato_mosaic_virus', 37:'Tomato_healthy'}

model = load_model('model_main.h5')

def predict_label(img_path):
    i=image.image_utils.load_img(img_path, target_size=(230,230))
    i = image.img_to_array(i)
    i=np.expand_dims(i, axis=0)
    i = i.reshape(1, 230,230,3)
    pred= np.argmax(model.predict(i), axis=1)
    return dic[pred]


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("predcit.html")

@app.route("/about")
def about_page():
    return "HI Man >:"

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['file']

        img_path = "static/" + img.filename
        print(request.files)
        img.save(img_path)

        p = predict_label(img_path)
        return render_template("predcit.html", prediction = p, img_path = img_path)
    return render_template("predcit.html")


if __name__ =='__main__':
    #app.debug = True
    app.run()