import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

from models import PrunableNet
from utils import compute_sparsity

st.title("Self-Pruning Neural Network")

model = PrunableNet()
model.load_state_dict(torch.load("outputs/checkpoints/model.pth", map_location="cpu"))
model.eval()

classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

uploaded_file = st.file_uploader("Upload an image", type=["png","jpg","jpeg"])

transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_container_width=True)

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1).numpy()[0]
        pred = np.argmax(probs)

    st.write("Prediction:", classes[pred])
    st.write("Confidence:", probs[pred])

st.write("Sparsity:", compute_sparsity(model))
