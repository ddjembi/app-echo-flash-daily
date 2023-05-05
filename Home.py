import streamlit as st
from PIL import Image

@st.cache_resource
def load_image(image_file):
    img = Image.open(image_file)
    return img

image = load_image('./images/llm_overview_services.png')  

"""
# Large Language Models in action

Large language models are sophisticated artificial intelligence tools that are capable of understanding and processing human language.

They can be used for a wide range of applications, including natural language processing, content creation, virtual assistants, information retrieval, and data analysis.

These tools have the potential to transform many industries and improve the way we communicate, search for information, and interact with technology. As the technology continues to evolve, we can expect to see even more innovative applications of large language models in the future.

"""

st.image(image)