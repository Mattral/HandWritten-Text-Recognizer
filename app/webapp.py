import os
import cv2
import numpy as np
from PIL import Image
from path import Path
import streamlit as st
from typing import Tuple
from dataloader_iam import Batch
from model import Model, DecoderType
from preprocessor import Preprocessor
from streamlit_drawable_canvas import st_canvas


def get_img_size(line_mode: bool = False) -> Tuple[int, int]:
    """
    Auxiliary method that sets the height and width
    Height is fixed while width is set according to the Model used.
    """
    if line_mode:
        return 256, get_img_height()
    return 128, get_img_height()

def get_img_height() -> int:
    """
    Auxiliary method that sets the height, which is fixed for the Neural Network.
    """
    return 32

def infer(line_mode: bool, model: Model, fn_img: Path) -> None:
    """
    Auxiliary method that does inference using the pretrained models:
    Recognizes text in an image given its path.
    """
    img = cv2.imread(fn_img, cv2.IMREAD_GRAYSCALE)
    assert img is not None

    preprocessor = Preprocessor(get_img_size(line_mode), dynamic_width=True, padding=16)
    img = preprocessor.process_img(img)

    batch = Batch([img], None, 1)
    recognized, probability = model.infer_batch(batch, True)
    return [recognized, probability]

def main():

    #Website properties
    st.set_page_config(
        page_title = "HTR App",
        page_icon = ":pencil:",
        layout = "centered",
        initial_sidebar_state = "auto",
    )

    st.title('HTR Simple Application')
    
    st.markdown("""
    Streamlit Web Interface for Handwritten Text Recognition (HTR), implemented with TensorFlow and trained on the IAM off-line HTR dataset. The model takes images of single words or text lines (multiple words) as input and outputs the recognized text. 
    """, unsafe_allow_html=True)
    
    st.markdown("""
    Predictions can be made using one of two models:
    - [Model 1](https://www.dropbox.com/s/mya8hw6jyzqm0a3/word-model.zip?dl=1) (Trained on Single Word Images) 
    - [Model 2](https://www.dropbox.com/s/7xwkcilho10rthn/line-model.zip?dl=1) (Trained on Text Line Images)    
    """, unsafe_allow_html=True)

    st.subheader('Select a Model, Choose the Arguments and Draw in the box below or Upload an Image to obtain a prediction.')

    #Selectors for the model and decoder
    modelSelect = st.selectbox("Select a Model", ['Single_Model', 'Line_Model'])

    decoderSelect = st.selectbox("Select a Decoder", ['Bestpath', 'Beamsearch', 'Wordbeamsearch'])

    #Mappings (dictionaries) for the model and decoder. Asigns the directory or the DecoderType of the selected option.
    modelMapping = {
        "Single_Model": '../model/word-model',
        "Line_Model": '../model/line-model'
    }

    decoderMapping = {
        'Bestpath': DecoderType.BestPath,
        'Beamsearch': DecoderType.BeamSearch,
        'Wordbeamsearch': DecoderType.WordBeamSearch
    }

    #Slider for pencil width
    strokeWidth = st.slider("Stroke Width: ", 1, 25, 6)

    #Canvas/Text Box for user input. BackGround Color must be white (#FFFFFF) or else text will not be properly recognised.
    inputDrawn = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)", 
        stroke_width=strokeWidth,
        update_streamlit=True,
        height = 200,
        width = 400,
        drawing_mode='freedraw',
        key="canvas",
        background_color = '#FFFFFF'
    )

    #Buffer for user input (images uploaded from the user's device)
    inputBuffer = st.file_uploader("Upload an Image", type=["png"])

    #Infer Button
    inferBool = st.button("Recognize Word")

    #We start infering once we have the user input and he presses the Infer button.
    if ((inputDrawn.image_data is not None or inputBuffer is not None) and inferBool == True):
        
        #We turn the input into a numpy array
        if inputDrawn.image_data is not None:
            inputArray = np.array(inputDrawn.image_data)
        
        if inputBuffer is not None:
            inputBufferImage = Image.open(inputBuffer)
            inputArray = np.array(inputBufferImage)

        #We turn this array into a .png format and save it. 
        inputImage = Image.fromarray(inputArray.astype('uint8'), 'RGBA')
        inputImage.save('userInput.png')
        #We obtain the model directory and the decoder type from their mapping
        modelDir = modelMapping[modelSelect]
        decoderType = decoderMapping[decoderSelect]

        #Finally, we call the model with this image as attribute and display the Best Candidate and its probability on the Interface
        model = Model(list(open(modelDir + "/charList.txt").read()), modelDir, decoderType, must_restore=True)
        inferedText = infer(modelDir == '../model/line-model', model, 'userInput.png')

        st.write("**Best Candidate: **", inferedText[0][0])
        st.write("**Probability: **", str(inferedText[1][0]*100) + "%")

if __name__ == "__main__":
    main()
