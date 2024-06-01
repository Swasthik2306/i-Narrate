import streamlit as st
import numpy as np
from PIL import Image
import io
from keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

from tensorflow.keras.models import Model

#____________________________________________

#----------------BRAILLE-------------------
from pybraille import convertText

#----------------BRAILLE DOWNLOAD FUNCTION-
from pdf1 import generate_pdf
from pdf5 import generate_merged_pdf
#____________________________________________


#----------------AUDIO---------------------
from gtts import gTTS
#____________________________________________


#____________________________________________
#----------------STORY GENERATION------------
from story import generate_story_from_caption
#____________________________________________



st.set_page_config(page_title="I-Narrate")


st.title('I-Narrate')

uploaded_file = st.file_uploader("Choose a image file", type=['png', 'jpg', 'jpeg'])

max_length = 35
model = load_model('model25.h5')
vgg_model = VGG16()
# restructure the model
vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)


# generate caption for an image

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word

    return None

def predict_caption(model, image, tokenizer, max_length):

    # add start tag for generation process
    in_text = 'startseq'

    # iterate over the max length of sequence
    for i in range(max_length):

        # encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad the sequence
        sequence = pad_sequences([sequence], max_length)
        # predict next word
        yhat = model.predict([image, sequence], verbose=0)
        # get index with high probability
        yhat = np.argmax(yhat)
        # convert index to word
        word = idx_to_word(yhat, tokenizer)
        # stop if word not found
        if word is None:
            break
        # append word as input for generating next word
        in_text += " " + word
        # stop if we reach end tag
        if word == 'endseq':
            break

    return in_text





if uploaded_file is not None:
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

    # Extract features
    feature = vgg_model.predict(image_array, verbose=0)

    # Predict from the trained model
    predicted_value = predict_caption(model, feature, tokenizer, max_length)
    predicted_value = " ".join(predicted_value.split()[1:-1])
    
    # braille conversion
    braille=convertText(predicted_value)
    
    
    predicted_value=generate_story_from_caption(predicted_value)
    braille=convertText(predicted_value)
    
    
    
    # voice conversion
    tts = gTTS(predicted_value)   
    tts.save('audio.mp3') 
    

    # Resize the original image to (360, 360)
    opencv_image = image.resize((360, 360))

    # Display the resized image
    st.image(opencv_image, channels="RGB")
    
    
#______________________<--CAPTIONING-->__________________________
    # Display the predicted caption
    st.header("Predicted Story is")
    st.write(predicted_value)
#________________________________________________________________
    
    
#______________________<--BRAILLE-->_____________________________
    st.write(braille)
    
    #pdf_data=generate_pdf(predicted_value)
    #st.download_button("Download PDF", data=pdf_data, file_name="i-narrate_braille_script.pdf")
    
    input_pdf_path = "Doc1.pdf"
    text = predicted_value
    position = (100, 470)  # Position where the text will appear
    image_path = uploaded_file
    image_width = 200
    image_height = 200

    merged_pdf = generate_merged_pdf(input_pdf_path, text, position, image_path, image_width, image_height)

# Write the merged PDF to a file
    merged_pdf_bytes = io.BytesIO()
    merged_pdf.write(merged_pdf_bytes)
    merged_pdf_bytes.seek(0)

# Download the merged PDF
    st.download_button("Download PDF", data=merged_pdf_bytes, file_name="merged.pdf")
#________________________________________________________________
    
#______________________<--AUDIO->________________________________

    st.audio("audio.mp3")
#________________________________________________________________


