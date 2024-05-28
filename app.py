import numpy as np
import torch
import os
import torchvision
import streamlit as st
from PIL import Image
from fractureModel import FractureModel


model=FractureModel(input_shape=3,output_shape=1,hidden_units=8)
model.load_state_dict(torch.load('fracture-detection_model.pth'))

image_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
])

classes=['fractured', 'not fractured']
model.eval()


def predict(image):
    with torch.inference_mode():
        image = Image.open(image)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image_transforms(image).unsqueeze(0)
        output = model.predict(image)
        return output


st.title("Fracture Detection")
st.write("This is a web app to detect fractures in X-ray images.")
st.write("Upload an image of an X-ray and the app will tell you if the bone is fractured or not.")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
print(uploaded_file)
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', width=200)

    # st.write("Classifying...")
    with st.spinner('Wait for it...'):
        output = predict(uploaded_file)
        prediction = classes[torch.round(output).int().item()]
        st.write(f"Prediction: {prediction}")
        Probablity = 1-output.item() 
        st.write(f"Fractured: {np.round(Probablity*100, 3)}%")

