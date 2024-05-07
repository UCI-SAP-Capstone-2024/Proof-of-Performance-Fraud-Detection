import streamlit as st
from PIL import Image
import io
import docx
from main import main

# Setting the page config with a specific title
st.set_page_config(page_title="Document and Image Analysis", layout="wide")

# Styling using HTML and CSS
st.markdown("""
<style>
.stApp {
    background-color: #fafafa;
}
h1 {
    color: #3366ff;
}
</style>
""", unsafe_allow_html=True)

st.title('Document and Image Analysis')

col1, col2 = st.columns(2)

with col1:
    st.header("Upload a DOCX file")
    uploaded_file_doc = st.file_uploader("Choose a DOCX file", type=['docx'], key="file_uploader_doc")

with col2:
    st.header("Upload an Image")
    uploaded_file_img = st.file_uploader("Choose an Image", type=['jpg', 'png', 'jpeg'], key="file_uploader_img")

if uploaded_file_doc is not None and uploaded_file_img is not None:
    # Read DOCX file
    doc = docx.Document(io.BytesIO(uploaded_file_doc.getvalue()))

    # Read Image
    image = Image.open(io.BytesIO(uploaded_file_img.getvalue()))

    # Process files and generate result
    result = main(doc, image)

    # Display the result
    st.write("Result from processing your files:", result)

st.write("Upload a DOCX file and an image to see the result of processing.")