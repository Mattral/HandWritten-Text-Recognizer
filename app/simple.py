import cv2
import numpy as np
from pathlib import Path
from model import Model, DecoderType
from preprocessor import Preprocessor
from dataloader_iam import Batch

import tensorflow as tf

def get_img_size(line_mode: bool = False) -> tuple[int, int]:
    """
    Auxiliary method that sets the height and width.
    Height is fixed while width is set according to the Model used.
    """
    if line_mode:
        return 256, get_img_height()
    return 128, get_img_height()

def get_img_height() -> int:
    """
    Auxiliary method that sets the fixed height for the Neural Network.
    """
    return 32

def infer(line_mode: bool, model: Model, fn_img: str) -> None:
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
    return recognized, probability

def main(image_path: str, model_path: str, decoder_type: DecoderType):
    """
    Main function to load the model, perform inference on the input image,
    and print the result.
    """
    # Load the model
    char_list_path = model_path + "/charList.txt"
    model = Model(list(open(char_list_path).read()), model_path, decoder_type, must_restore=True)
    
    # Perform inference
    recognized, probability = infer(model_path.endswith('line-model'), model, image_path)
    
    # Print the results
    print("Recognized Text:", recognized[0])
    print("Probability:", probability[0])

if __name__ == "__main__":
    # Example usage
    # Define the image path, model directory, and decoder type here
    image_path = 'word.png'  # Update this path
    model_path = '../model/word-model'  # or '../model/line-model' depending on your model
    decoder_type = DecoderType.BestPath  # Change as needed: BestPath, BeamSearch, WordBeamSearch

    # Call the main function with the specified parameters
    main(image_path, model_path, decoder_type)
