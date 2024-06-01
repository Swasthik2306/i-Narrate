from flask import Flask, render_template, request, send_from_directory, send_file
import numpy as np
import io
import os
from PIL import Image
from keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from keras.models import load_model
from tensorflow.keras.models import Model
#--------------------------PYBRAILLE--------------------------------------
from pybraille import convertText
#-------------------------------------------------------------------------


#--------------------------TEXT TO SPEECH :gTTS---------------------------
from gtts import gTTS
#-------------------------------------------------------------------------

#--------------------------DEFINED FUNCTIONS------------------------------
from app1 import predict_caption
from story import generate_story_from_caption
from pdf5 import generate_merged_pdf
#-------------------------------------------------------------------------

import base64

#--------------------------PDF LAYOUT-------------------------------------
image_width = 200
image_height = 200
input_pdf_path = "static/Doc1.pdf"
position = (100, 470)
#-------------------------------------------------------------------------

uploaded_file=None
story=None
#--------------------------FLASK APPLICATION------------------------------
app = Flask(__name__)
#-------------------------------------------------------------------------

#--------------------------LOAD REQUIRED MODEL----------------------------
max_length = 35
model = load_model('model25.h5')
vgg_model = VGG16()
# restructure the model
vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
#-------------------------------------------------------------------------

#--------------------------IMAGE PREPROCESSING AND FEATURE EXTRACTION-----
def preprocess_image(uploaded_file):
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = Image.open(io.BytesIO(file_bytes))

    # Resize the image to (224, 224)
    resized_image = image.resize((224, 224))

    # Convert the resized image to a NumPy array
    image_array = np.array(resized_image)

    # Reshape data for the model
    image_array = image_array.reshape((1, image_array.shape[0], image_array.shape[1], image_array.shape[2]))

    # Preprocess image for VGG
    image_array = preprocess_input(image_array)

    feature = vgg_model.predict(image_array, verbose=0)

    return feature
#---------------------------------------------------------------------------

#--------------------------TO HOME PAGE-------------------------------------
@app.route('/')
def index():
    return render_template('index.html')
#---------------------------------------------------------------------------

#--------------------------TO i-NARRATE PAGE-------------------------------------
@app.route('/i-narrate', methods=['GET', 'POST'])
def inarrate():
    img_data = None
    global story, uploaded_file, image_path
    
    braille = None
    tts = None
    merged_pdf=None

    if request.method == 'POST':
        # Get the uploaded image file
        uploaded_file = request.files['image']
        
        # Check if the image is not in JPG format
        if uploaded_file.filename.split('.')[-1] != 'jpg':
            # Convert the image to JPG format
            img = Image.open(uploaded_file)
            img_jpg = img.convert('RGB')
            uploaded_file = io.BytesIO()
            img_jpg.save(uploaded_file, 'JPEG')
            uploaded_file.seek(0)
        # Read the uploaded file content
        

        # Read the uploaded file content
        img_bytes = uploaded_file.read()

        # Preprocess the image
        feature = preprocess_image(io.BytesIO(img_bytes))

        # Predict the caption for the image
        caption = predict_caption(model, feature, tokenizer, max_length)
        caption = " ".join(caption.split()[1:-1])
        
        #--------------------------GENERATE STORY--------------------------------------
        story=generate_story_from_caption(caption)
        #------------------------------------------------------------------------------
        
        #--------------------------CONVERT STORY TO BRAILLE SCRIPT---------------------
        braille=convertText(story)
        #------------------------------------------------------------------------------
        
        #--------------------------CONVERT STORY TO AUDIBLE FILE-----------------------
        tts = gTTS(story)   
        tts.save('static/audio.mp3') 
        #------------------------------------------------------------------------------
        
        #--------------------------GENERATE PDF----------------------------------------
        text = story
        image_path = uploaded_file
        merged_pdf = generate_merged_pdf(input_pdf_path, text, position, image_path, image_width, image_height)
        #------------------------------------------------------------------------------
        
        # Convert the uploaded file content to a base64 string
        img_data = base64.b64encode(img_bytes).decode('utf-8')
        

    return render_template('i-narrate.html', img_data=img_data, caption_data=story, braille=braille, audio_file='audio.mp3', pdf_file=merged_pdf)
#----------------------------------------------------------------------------------------------------------------

#--------------------------TO STORY PAGE------------------------------------------
@app.route('/story', methods=['GET', 'POST'])
def storyd():
    img_data = None
    global story, uploaded_file, image_path
    
    braille = None
    tts = None
    merged_pdf=None

    if request.method == 'POST':
        # Get the uploaded image file
        uploaded_file = request.files['image']
            
        # Check if the image is not in JPG format
        if uploaded_file.filename.split('.')[-1] != 'jpg':
            # Convert the image to JPG format
            img = Image.open(uploaded_file)
            img_jpg = img.convert('RGB')
            uploaded_file = io.BytesIO()
            img_jpg.save(uploaded_file, 'JPEG')
            uploaded_file.seek(0)
            
            
        # Read the uploaded file content
        img_bytes = uploaded_file.read()

        # Preprocess the image
        feature = preprocess_image(io.BytesIO(img_bytes))

        # Predict the caption for the image
        caption = predict_caption(model, feature, tokenizer, max_length)
        caption = " ".join(caption.split()[1:-1])
        
        #--------------------------GENERATE STORY--------------------------------------
        story=generate_story_from_caption(caption)
        #------------------------------------------------------------------------------
        
        #--------------------------GENERATE PDF----------------------------------------
        text = story
        image_path = uploaded_file
        merged_pdf = generate_merged_pdf(input_pdf_path, text, position, image_path, image_width, image_height)
        #------------------------------------------------------------------------------

        # Convert the uploaded file content to a base64 string
        img_data = base64.b64encode(img_bytes).decode('utf-8')
        

    return render_template('story.html', img_data=img_data, caption_data=story, pdf_file=merged_pdf)
#-----------------------------------------------------------------------------------------------------------

#--------------------------TO BRAILLE PAGE------------------------------------------
@app.route('/braille', methods=['GET', 'POST'])
def brailled():
    img_data = None
    global story, uploaded_file, image_path
    
    braille = None
    tts = None
    merged_pdf=None

    if request.method == 'POST':
        # Get the uploaded image file
        uploaded_file = request.files['image']
    
        # Check if the image is not in JPG format
        if uploaded_file.filename.split('.')[-1] != 'jpg':
            # Convert the image to JPG format
            img = Image.open(uploaded_file)
            img_jpg = img.convert('RGB')
            uploaded_file = io.BytesIO()
            img_jpg.save(uploaded_file, 'JPEG')
            uploaded_file.seek(0)

        # Read the uploaded file content
        img_bytes = uploaded_file.read()

        # Preprocess the image
        feature = preprocess_image(io.BytesIO(img_bytes))

        # Predict the caption for the image
        caption = predict_caption(model, feature, tokenizer, max_length)
        caption = " ".join(caption.split()[1:-1])
        
        #--------------------------GENERATE STORY--------------------------------------
        story=generate_story_from_caption(caption)
        #------------------------------------------------------------------------------
        
        #--------------------------GENERATE BRAILLE------------------------------------
        braille=convertText(story)
        #------------------------------------------------------------------------------
        
        #--------------------------GENERATE PDF---------------------------------------
        text = story
        image_path = uploaded_file
        merged_pdf = generate_merged_pdf(input_pdf_path, text, position, image_path, image_width, image_height)
        #------------------------------------------------------------------------------

        # Convert the uploaded file content to a base64 string
        img_data = base64.b64encode(img_bytes).decode('utf-8')
        

    return render_template('braille.html', img_data=img_data, braille=braille, pdf_file=merged_pdf)
#--------------------------PYBRAILLE------------------------------------------------------------------------------

#--------------------------TO AUDIO PAGE-----------------------------------------
@app.route('/audio', methods=['GET', 'POST'])
def audiod():
    img_data = None
    global story, uploaded_file, image_path
    
    braille = None
    tts = None
    merged_pdf=None

    if request.method == 'POST':
        # Get the uploaded image file
        uploaded_file = request.files['image']

        # Check if the image is not in JPG format
        if uploaded_file.filename.split('.')[-1] != 'jpg':
            # Convert the image to JPG format
            img = Image.open(uploaded_file)
            img_jpg = img.convert('RGB')
            uploaded_file = io.BytesIO()
            img_jpg.save(uploaded_file, 'JPEG')
            uploaded_file.seek(0)

        # Read the uploaded file content
        img_bytes = uploaded_file.read()

        # Preprocess the image
        feature = preprocess_image(io.BytesIO(img_bytes))

        # Predict the caption for the image
        caption = predict_caption(model, feature, tokenizer, max_length)
        caption = " ".join(caption.split()[1:-1])
        
        #--------------------------GENERATE STORY--------------------------------------
        story=generate_story_from_caption(caption)
        #------------------------------------------------------------------------------
        
        #--------------------------CONVERT STORY T0 AUDIBLE FILE-----------------------
        tts = gTTS(story)   
        tts.save('static/audio.mp3') 
        #-------------------------------------------------------------------------------
        
        #--------------------------GENERATE PDF--------------------------------------
        text = story
        image_path = uploaded_file
        merged_pdf = generate_merged_pdf(input_pdf_path, text, position, image_path, image_width, image_height)
        
        # Convert the uploaded file content to a base64 string
        img_data = base64.b64encode(img_bytes).decode('utf-8')
        

    return render_template('audio.html', img_data=img_data, audio_file='audio.mp3', pdf_file=merged_pdf)
#---------------------------------------------------------------------------------------------------------------

#--------------------------DOWNLOAD FULL PDF WITH BRAILLE--------------------------------------
@app.route('/download_pdf', methods=['GET'])
def download_pdf():
    pdf_file_path = "static/final.pdf"
    return send_file(pdf_file_path, as_attachment=True, download_name='i_narrate_braille.pdf')
#----------------------------------------------------------------------------------------------

#--------------------------DOWNLOAD STORY PDF--------------------------------------------------
@app.route('/download_story_pdf', methods=['GET'])
def download_story_pdf():
    pdf_file_path = "static/inarrate.pdf"
    return send_file(pdf_file_path, as_attachment=True, download_name='i_narrate_story.pdf')
#----------------------------------------------------------------------------------------------

if __name__ == '__main__':
    app.run(debug=True)