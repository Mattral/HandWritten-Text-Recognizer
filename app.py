import os
import cv2
import numpy as np
from PIL import Image
from path import Path
import streamlit as st
from typing import Tuple
import easyocr  # Import EasyOCR

from pathlib import Path
import sys

# Add the 'app' directory to the sys.path
# Assuming 'app' is in the current working directory
sys.path.append(str(Path(__file__).parent / 'app'))
from app.dataloader_iam import Batch
from app.model import Model, DecoderType
from app.preprocessor import Preprocessor
from streamlit_drawable_canvas import st_canvas


# Set page config at the very beginning (only executed once)
st.set_page_config(
    page_title="HTR App",
    page_icon=":pencil:",
    layout="centered",
    initial_sidebar_state="auto",
)

ms = st.session_state
if "themes" not in ms: 
  ms.themes = {"current_theme": "light",
                    "refreshed": True,
                    
                    "light": {"theme.base": "dark",
                              "theme.backgroundColor": "black",
                              "theme.primaryColor": "#c98bdb",
                              "theme.secondaryBackgroundColor": "#5591f5",
                              "theme.textColor": "white",
                              "theme.textColor": "white",
                              "button_face": "🌜"},

                    "dark":  {"theme.base": "light",
                              "theme.backgroundColor": "white",
                              "theme.primaryColor": "#5591f5",
                              "theme.secondaryBackgroundColor": "#82E1D7",
                              "theme.textColor": "#0a1464",
                              "button_face": "🌞"},
                    }
  

def ChangeTheme():
  previous_theme = ms.themes["current_theme"]
  tdict = ms.themes["light"] if ms.themes["current_theme"] == "light" else ms.themes["dark"]
  for vkey, vval in tdict.items(): 
    if vkey.startswith("theme"): st._config.set_option(vkey, vval)

  ms.themes["refreshed"] = False
  if previous_theme == "dark": ms.themes["current_theme"] = "light"
  elif previous_theme == "light": ms.themes["current_theme"] = "dark"


btn_face = ms.themes["light"]["button_face"] if ms.themes["current_theme"] == "light" else ms.themes["dark"]["button_face"]
st.button(btn_face, on_click=ChangeTheme)

if ms.themes["refreshed"] == False:
  ms.themes["refreshed"] = True
  st.rerun()


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

def infer_super_model(image_path) -> None:
    reader = easyocr.Reader(['en'])  # Initialize EasyOCR reader
    result = reader.readtext(image_path)
    recognized_texts = [text[1] for text in result]  # Extract recognized texts
    probabilities = [text[2] for text in result]  # Extract probabilities
    return recognized_texts, probabilities



def main():

    st.title('Extract text from Image Demo')
    
    st.markdown("""
    Streamlit Web Interface for Handwritten Text Recognition (HTR), Optical Character Recognition (OCR) 
                implemented with TensorFlow and trained on the IAM off-line HTR dataset. 
                The model takes images of single words or text lines (multiple words) as input and outputs the recognized text. 
    """, unsafe_allow_html=True)
    
    st.markdown("""
    Predictions can be made using one of two models:
    - Single_Model (Trained on Single Word Images) 
    - Line_Model (Trained on Text Line Images)    
    - Super_Model ( Most Robust Option for English )
    - Burmese (Link)
    """, unsafe_allow_html=True)

    st.subheader('Select a Model, Choose the Arguments and Draw in the box below or Upload an Image to obtain a prediction.')

    #Selectors for the model and decoder
    modelSelect = st.selectbox("Select a Model", ['Single_Model', 'Line_Model', 'Super_Model'])
    

    if modelSelect != 'Super_Model':
        decoderSelect = st.selectbox("Select a Decoder", ['Bestpath', 'Beamsearch', 'Wordbeamsearch'])


    #Mappings (dictionaries) for the model and decoder. Asigns the directory or the DecoderType of the selected option.
    modelMapping = {
        "Single_Model": 'model/word-model',
        "Line_Model": 'model/line-model'
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
        background_image=None,
        height = 200,
        width = 400,
        drawing_mode='freedraw',
        key="canvas",
        background_color = '#FFFFFF'
    )

    #Buffer for user input (images uploaded from the user's device)
    inputBuffer = st.file_uploader("Upload an Image", type=["png"])

    #Inference Button
    inferBool = st.button("Recognize Text")

    # After clicking the "Recognize Text" button, check if the model selected is Super_Model
    if inferBool:
        if modelSelect == 'Super_Model':
            inputArray = None  # Initialize inputArray to None

            # Handling uploaded file
            if inputBuffer is not None:
                with Image.open(inputBuffer).convert('RGB') as img:
                    inputArray = np.array(img)

            # Handling canvas data
            elif inputDrawn.image_data is not None:
                # Convert RGBA to RGB
                inputArray = cv2.cvtColor(np.array(inputDrawn.image_data, dtype=np.uint8), cv2.COLOR_RGBA2RGB)

            # Now check if inputArray has been set
            if inputArray is not None:
                # Initialize EasyOCR Reader
                reader = easyocr.Reader(['en'])  # Assuming English language; adjust as necessary
                # Perform OCR
                results = reader.readtext(inputArray)

                # Display results
                all_text = ''
                for (bbox, text, prob) in results:
                    all_text += f'{text} (confidence: {prob:.2f})\n'

                st.write("**Recognized Texts and their Confidence Scores:**")
                st.text(all_text)
            else:
                st.write("No image data found. Please upload an image or draw on the canvas.")


        else:
            # Handle other model selections as before
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
