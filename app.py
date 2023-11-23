from flask import Flask, render_template, request, redirect, url_for
import os
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import base64
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.applications.xception import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.xception import Xception
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load the saved model
saved_model_path = "models/model_13.keras"
loaded_model = load_model(saved_model_path)

# Load the features and tokenizer
with open("features.p", "rb") as f:
    features = pickle.load(f)

with open("tokenizer.p", "rb") as f:
    tokenizer = pickle.load(f)

# Set max_length
max_length = 32

# Function to extract features from a new image
def extract_features(filename):
    model = Xception(include_top=False, pooling='avg')
    image = load_img(filename, target_size=(299, 299))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    feature = model.predict(image)
    return feature

def generate_caption(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = word_for_id(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq' or word == 'end':
            break

    # Remove the 'startseq' and 'endseq' tokens from the generated caption
    caption = in_text.split()[1:-1]
    caption = ' '.join(caption)
    return caption

# Function to map an integer to its corresponding word
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST' and 'image' in request.files:
        # Handle image upload
        uploaded_image = request.files['image']
        if uploaded_image.filename != '':
            # Save the uploaded image
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_image.filename)
            uploaded_image.save(image_path)

            # Generate caption for the uploaded image
            photo = extract_features(image_path)
            caption = generate_caption(loaded_model, tokenizer, photo, max_length)

            # Capitalize the first letter of the predicted caption
            caption = caption.capitalize()

            # Encode the image as base64
            with open(image_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

            # Display the image and caption on a new webpage
            return render_template('result.html', encoded_image=encoded_image, caption=caption)

    # Render the homepage with an image upload form
    return render_template('home.html')


@app.route('/home')
def return_home():
    return redirect(url_for('home'))

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
