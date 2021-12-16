import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
import streamlit as st  #Web App
#st.title("Cross Modal Machine Learning")
from PIL import Image as imm #Image Processing
from PIL import ImageOps
import numpy as np #Image Processing
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow import keras
import pickle
import base64

LOGO_IMAGE = "logo.png"

st.markdown(
    """
    <style>
    .container {
        display: flex;
        background-image: url('logo.jpg');
    }
    .logo-text {
        font-weight:700 !important;
        font-size:20px !important;
        color: #f64133 !important;
        padding-top: 75px !important;
    }
    .logo-img {
        float:center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <div class="container">
        <img class="logo-img" src="data:image/png;base64,{base64.b64encode(open(LOGO_IMAGE, "rb").read()).decode()}">
    </div>
    """,
    unsafe_allow_html=True
)
#from IPython.display import Image 
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
texts = pickle.load(open('text.pkl', 'rb'))
resnet_model = ResNet50(weights='imagenet', include_top=False)
image_model = keras.models.load_model("./10000-model.image", compile=False)
caption_representations = np.load('./caption-test-1000-representations.npy')
def extract_features(file):
    size = (224,224)    
    file2= ImageOps.fit(file, size, imm.ANTIALIAS)
    x = np.asarray(file2)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = resnet_model.predict(x)
    return np.expand_dims(features.flatten(), axis=0)

def generate_caption(image_filename, n=5):
    image_representation = image_model.predict(extract_features(image_filename))
    scores = np.dot(caption_representations, image_representation.T).flatten()
    indices = np.argpartition(scores, -n)[-n:]
    indices = indices[np.argsort(scores[indices])]
    st.markdown(
    f"""
    <div class="container">
        <p class="logo-text">Top Results are:</p>
    </div>
    """,
    unsafe_allow_html=True
)
    for i in [int(x) for x in reversed(indices)]:
        st.subheader(str(texts[i]))

    
#subtitle
st.markdown("## Cross Modal Machine learning : Image Annotation ")

st.markdown("")

#image uploader
file = st.file_uploader(label = "Upload your image here",type=['png','jpg','jpeg'])


if file is not None:

    input_image = imm.open(file) #read image
    st.image(input_image) #display image

    with st.spinner("🤖 AI is at Work! "):
        generate_caption(input_image)
        #st.write("done")
    #st.success("Here you go!")
    st.balloons()
#else:
    #st.write("Upload an Image")

st.caption("Made with ❤")





