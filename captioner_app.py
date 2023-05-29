# @author: Rohitaswa Sarbhangia
# Resources: HuggingFace nlpconnect/vit-gpt2-image-captioning model
# Here I'm using a pretrained model for image captioning. Finally using streamlit 
# to demonstrate where a user can upload an image and the model will give the caption. 


# Importing necessary packages
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
import streamlit as st


st.set_option('deprecation.showfileUploaderEncoding', False)  # This supresses warnings
@st.cache_resource  # This caches the ML model so that the page is faster to load
def load_model():
  # Loading the pretrained model and encoder
  model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
  feat = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
  tok = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
  return model, feat, tok

model, feature_extractor, tokenizer = load_model()


st.header("Image Captioner 🖼")    # Header

# Using GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

# Making the prediction
def predict_step(image_paths):
  images = []
  for i_image in image_paths:   # Looping through images if multiple images uploaded
    st.image(i_image)   # Displaying the image
    # i_image = image_path
    if i_image.mode != "RGB":    
      i_image = i_image.convert(mode="RGB")

    images.append(i_image)

  pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
  pixel_values = pixel_values.to(device)

  output_ids = model.generate(pixel_values, **gen_kwargs)   # Running the model
 
  preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)  # Getting the predictions
  preds = [pred.strip() for pred in preds]
  return preds

file = st.file_uploader("Please upload an image", type=["jpg", "png"])   # Uploading the file

if file is None:   # if no file uploaded
  st.write("Please upload a file")
else:
  # st.write(file.name)
  # st.write(type(file.name))
  image_file = Image.open(file)  # Reading the file
  st.write("The image is about ", predict_step([image_file])[0]) # printing the caption
