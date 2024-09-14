import time
import torch
from diffusers import StableDiffusionPipeline
import streamlit as st
from PIL import Image
import io
import base64

# Set mixed precision and cuDNN auto-tuner
torch.set_default_dtype(torch.float16)
torch.backends.cudnn.benchmark = True 

# Load the saved Stable Diffusion model
model_load_path = "temp_extracted_model"  # Path to your saved model
pipe = StableDiffusionPipeline.from_pretrained(model_load_path, torch_dtype=torch.float16)
pipe.to("cuda")

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover;
        background-position: center;
    }}
    .stApp::before {{
        content: "";
        position: absolute;
        top: 0;
        right: 0;
        bottom: 0;
        left: 0;
        background-color: rgba(0, 0, 0, 0.5);
        z-index: -1;
    }}
    .main {{
        background-color: rgba(0, 0, 0, 0.7);
        padding: 20px;
        border-radius: 10px;
    }}
    h1, h2, h3, h4, h5, h6, p, div {{
        font-family: 'Arial', sans-serif;
        color: white;
    }}
    .sidebar .sidebar-content {{
        background: rgba(0, 0, 0, 0.7);
    }}
    .stButton>button {{
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 5px;
        transition: background-color 0.0s;
    }}
    .stButton>button:hover {{
        background-color: #45a049;
    }}
    .stTextInput>div>div>input {{
        background-color: rgba(255, 255, 255, 0.1);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.5);
    }}
    .stSelectbox>div>div>select {{
        background-color: rgba(255, 255, 255, 0.1);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.5);
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

# Use the function to set background
add_bg_from_local('background2.jpg')  # Replace with the path to your image

# Streamlit app title
st.title("Stable Diffusion Image Generator")

# Sidebar for user input
st.sidebar.header("Settings")
prompt = st.sidebar.text_input("Enter your prompt:", "flying cars and neon lights")
image_size = st.sidebar.selectbox("Select image size", ["256x256", "512x512", "1024x1024"], index=2)

# Convert image size to tuple
size_mapping = {
    "256x256": (256, 256),
    "512x512": (512, 512),
    "1024x1024": (1024, 1024)
}
size = size_mapping[image_size]

# Generate the image
if st.sidebar.button("Generate Image"):
    if prompt:
        with st.spinner("Generating image..."):
            start_time = time.time()
            with torch.no_grad():
                images = pipe(prompt, num_images=1, height=size[0], width=size[1]).images  # Generate one image
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # Display the generated image
            for idx, img in enumerate(images):
                st.image(img, caption="Generated Image", use_column_width=True)
            
            st.write(f"Time taken to generate the image: {elapsed_time:.2f} seconds")
            st.write(f"Image size: {image_size}")
    else:
        st.error("Please enter a prompt.")

# Information about the model
st.sidebar.write("### About this app")
st.sidebar.write(
    "This app uses a pre-trained Stable Diffusion model to generate images based on textual prompts. "
    "You can adjust settings such as the image size. "
    "Please provide a prompt to start the image generation process."
)

# Streamlit footer
st.sidebar.markdown("---")
st.sidebar.write("### Developed with ❤️ by Darshan Jebaraj")