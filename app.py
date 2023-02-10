from flask import Flask, render_template, request
import tensorflow as tf
import pickle
from matplotlib import image as matimg
app = Flask(__name__)
model = tf.keras.models.load_model('static/new_m.h5')
scaler = pickle.load(open('static/scaler.pkl', 'rb'))



model2 = tf.keras.models.load_model('static/pest_detector.h5')

@app.route('/')
def home():
    return render_template("index.html")


@app.route('/croprecom', methods = ['GET', 'POST'])
def crop_recom():
    if request.method == 'POST':
        try:

            input_str = list(request.form.values())
            input = [float(x) for x in input_str]
            input = tf.expand_dims(input, 0)
            input = scaler.transform(input)
            out = model.predict(input)
            print(out)
            crops = ['apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee',
        'cotton', 'grapes', 'jute', 'kidneybeans', 'lentil', 'maize',
        'mango', 'mothbeans', 'mungbean', 'muskmelon', 'orange', 'papaya',
        'pigeonpeas', 'pomegranate', 'rice', 'watermelon']
            
            ind = out[0].argmax()
            res = crops[ind]

            return render_template('result.html', result = res)
        #  return f"Your Crop Recommendatio is {res}
        except ValueError:
            return render_template('error.html')


    return render_template('crop_recom.html')

@app.route('/pest', methods = ['GET', 'POST'])
def pest_detect():
    if request.method == 'POST':
        try:
            f = request.files['file']
            img = matimg.imread(f)
            img = tf.image.resize(img, (240, 240))
            ind = model2.predict(tf.expand_dims(img, 0))[0].argmax()
            pests = ['aphids', 'armyworm', 'beetle', 'bollworm', 'grasshopper', 'mites', 'mosquito', 'sawfly', 'stem_borer']
            res = pests[ind]
            

            return render_template('res_pest.html', result = res)

        except ValueError:
            return render_template('error_pest.html')


    return render_template('pest.html')
    




if __name__ == '__main__':
    app.run(debug = True)