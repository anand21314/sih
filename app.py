# app.py

import streamlit as st
from PIL import Image
import torch
import timm
from torchvision import transforms
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from gtts import gTTS
import io

# --------------------
# 1. Model Setup
# --------------------
num_classes = 12
model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=num_classes)
model_path = "efficientb0_breed_classifier.pth"
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# --------------------
# 2. Image Preprocessing
# --------------------
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --------------------
# 3. Prediction Function
# --------------------
def predict(image):
    image = image.convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    return predicted.item(), confidence.item(), probabilities, image_tensor

# --------------------
# 4. Grad-CAM Function
# --------------------
def generate_gradcam(image_tensor, target_class):
    target_layers = [model.conv_head]
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=image_tensor.cpu())[0, :]
    rgb_img = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    rgb_img = np.clip((rgb_img * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406], 0, 1)

    # Overlay Grad-CAM exactly over image size
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    return visualization

# --------------------
# 5. Breed Labels + Info
# --------------------
breeds = [
    "Brown_Swiss", "Deoni", "Gir", "Holstein_Friesian", "Jaffrabadi", "Kangayam",
    "Kankrej", "Khillari", "Murrah", "Pandharpuri", "Sahiwal", "Toda"
]

breed_info_en = {
    "Brown_Swiss": "Known for high milk yield and docile temperament.",
    "Deoni": "Indigenous dual-purpose breed for milk and draught.",
    "Gir": "Famous for quality milk and resistance to disease.",
    "Holstein_Friesian": "Popular dairy breed worldwide with very high yield.",
    "Jaffrabadi": "Large buffalo breed used for both milk and draught.",
    "Kangayam": "Hardy draught breed from Tamil Nadu.",
    "Kankrej": "Dual-purpose breed with good milk and work ability.",
    "Khillari": "Strong draught breed found in Maharashtra.",
    "Murrah": "Buffalo breed famous for very high milk yield.",
    "Pandharpuri": "Indigenous cattle breed from Maharashtra.",
    "Sahiwal": "One of the best indigenous dairy breeds from Punjab.",
    "Toda": "Rare buffalo breed with unique appearance, from Nilgiris."
}

breed_info_te = {
    "Brown_Swiss": "బ్రౌన్ స్విస్: ఎక్కువ పాలను ఇస్తుంది మరియు నిబద్ధత కలిగి ఉంటుంది.",
    "Deoni": "డియోని: పాలు మరియు పని కోసం స్థానిక జాతి.",
    "Gir": "గిర్: ఉన్నత నాణ్యత పాలు మరియు వ్యాధి నిరోధకతతో ప్రసిద్ధి పొందింది.",
    "Holstein_Friesian": "హోల్స్టైన్ ఫ్రిజియన్: ప్రపంచవ్యాప్తంగా ప్రాచుర్యం పొందిన పాలు ఇస్తే జాతి.",
    "Jaffrabadi": "జఫ్రాబడి: పెద్ద ఆవు జాతి, పాలు మరియు పని కోసం.",
    "Kangayam": "కంగాయం: తమిళనాడులో మన్నింపు ఉన్న పని జాతి.",
    "Kankrej": "కంక్రేజ్: పాలు మరియు పని సామర్థ్యం కలిగిన ద్విపరి ఉపయోగ జాతి.",
    "Khillari": "ఖిల్లారి: మహారాష్ట్రలో దృఢమైన పని జాతి.",
    "Murrah": "ముర్రా: అధిక పాలను ఇస్తే జాతీయ జాబితా.",
    "Pandharpuri": "పంధర్పురి: మహారాష్ట్రలో స్థానిక ఆవు జాతి.",
    "Sahiwal": "సాహివాల్: పంచాబ్ లో అత్యుత్తమ స్థానిక పాలు ఇస్తే జాతి.",
    "Toda": "టోడా: నీలగిరుల్లో ప్రత్యేక రూపం కలిగిన అరుదైన జాబితా."
}

# --------------------
# 6. Text-to-Speech
# --------------------
def speak(text, lang='en'):
    tts = gTTS(text=text, lang=lang)
    audio_bytes = io.BytesIO()
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)
    st.audio(audio_bytes, format='audio/mp3')

# --------------------
# 7. Streamlit UI
# --------------------
st.set_page_config(page_title="GoNethra", page_icon="🐄", layout="wide")

# Green gradient background
st.markdown("""
    <style>
    .stApp {background: linear-gradient(to right, #a8e063, #56ab2f);}
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("📌 About GoNethra")
st.sidebar.write("AI-powered Cattle Breed Classifier with Grad-CAM and Voice Feature 🐄")
st.sidebar.markdown("---")
st.sidebar.write("👨‍💻 Developed for Smart India Hackathon")

# Language selection
language = st.sidebar.selectbox("Select Language | భాష ఎంచుకోండి", ["English", "Telugu"])

# Main content
if language == "English":
    st.title("🐄 GoNethra")
    st.subheader("AI-powered Cattle Breed Classifier with Grad-CAM & Voice")
else:
    st.title("🐄 గోనేత్ర")
    st.subheader("ఏఐ ఆధారిత పశు జాతి గుర్తింపు, Grad-CAM మరియు వాయిస్ ఫీచర్")

uploaded_file = st.file_uploader("📷 Upload an image | చిత్రాన్ని అప్‌లోడ్ చేయండి", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns([1, 1.5])

    # Prediction
    breed_index, confidence, probs, image_tensor = predict(image)
    predicted_breed = breeds[breed_index]

    with col1:
        st.image(image, caption='Uploaded Image', use_container_width=True)


    with col2:
        if language == "English":
            st.success(f"✅ Predicted Breed: **{predicted_breed}**")
            st.info(f"🔍 Confidence: {confidence*100:.2f}%")
            st.markdown(f"### 📖 About {predicted_breed}")
            st.write(breed_info_en[predicted_breed])
            st.markdown("### 📊 Top 3 Predictions")
        else:
            st.success(f"✅ గుర్తించిన జాతి: **{predicted_breed}**")
            st.info(f"🔍 నమ్మకం: {confidence*100:.2f}%")
            st.markdown(f"### 📖 {predicted_breed} గురించి")
            st.write(breed_info_te[predicted_breed])
            st.markdown("### 📊 అగ్ర 3 ఊహింపులు")

        # Top 3 predictions
        top3_idx = np.argsort(probs.squeeze().numpy())[-3:][::-1]
        for idx in top3_idx:
            prob_percent = probs[0, idx]*100
            st.write(f"- {breeds[idx]}: {prob_percent:.2f}%")

        # Grad-CAM exactly same size as input image
        gradcam_image = generate_gradcam(image_tensor, breed_index)
        if language == "English":
            st.markdown("### 🔥 Model Attention (Grad-CAM)")
        else:
            st.markdown("### 🔥 మోడల్ దృష్టి (Grad-CAM)")
        st.image(gradcam_image, use_column_width=True)

        # Speak button
        if st.button("🔊 Speak Breed Name | జాతి పేరు చెప్పు"):
            if language == "English":
                speak(f"The predicted breed is {predicted_breed}", lang='en')
            else:
                speak(predicted_breed, lang='te')

# Explore all breeds
st.markdown("---")
if language == "English":
    st.subheader("📚 Explore All Breeds")
else:
    st.subheader("📚 అన్ని జాతులను అన్వేషించండి")
selected_breed = st.selectbox("Choose a breed | జాతిని ఎంచుకోండి", breeds)
if language == "English":
    st.write(breed_info_en[selected_breed])
else:
    st.write(breed_info_te[selected_breed])
