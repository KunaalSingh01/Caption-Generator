import streamlit as st
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import google.generativeai as genai


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Caption Generator",
    page_icon="🖼️",
    layout="centered"
)

# ---------------- SIDEBAR ----------------
st.sidebar.title("📌 How to use")
st.sidebar.write("""
1️⃣ Upload an image  
2️⃣ Wait for AI to generate caption  
3️⃣ Gemini makes it creative ✨  
4️⃣ Copy & share 🎉  
""")

st.sidebar.markdown("---")
st.sidebar.info("⚠️ This is an AI-generated caption.\nFor creative use only.")

# ---------------- MAIN TITLE ----------------
st.markdown(
    "<h1 style='text-align: center;'>🖼️ AI Image Caption Generator</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center;'>Turn your images into <b>smart & creative captions</b> using AI ✨</p>",
    unsafe_allow_html=True
)

st.markdown("---")

# ---------------- LOAD GEMINI ----------------
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=GEMINI_API_KEY)
model_gemini = genai.GenerativeModel("gemini-1.5-flash")


# ---------------- LOAD BLIP MODEL (FIXED) ----------------
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base",
        torch_dtype=torch.float32
    )

    device = torch.device("cpu")  # Streamlit Cloud = CPU only
    model = model.to(device)

    return processor, model, device


processor, model, device = load_model()

# ---------------- GEMINI FUNCTION ----------------
def enhance_caption(raw_caption):
    prompt = f"""
You are an AI assistant.

Task:
1. What I see → Describe the image clearly in one simple sentence.
2. Caption for user → Write a creative, engaging caption with emojis and hashtags.

Rules:
- Do NOT repeat sentences word by word.
- Keep both parts different in tone and style.

Image description:
{raw_caption}

Output format:

What I see:
<description>

Caption for You:
<creative caption>
"""

    response = model_gemini.generate_content(prompt)
    return response.text.strip()


# ---------------- FILE UPLOAD ----------------
st.subheader("📤 Upload an Image")

uploaded_image = st.file_uploader(
    "Choose an image (jpg, png, jpeg)",
    type=["jpg", "png", "jpeg"]
)

if uploaded_image:
    image = Image.open(uploaded_image).convert("RGB")

    st.markdown("### 🖼️ Preview")
    st.image(image, use_container_width=True)

    with st.spinner("🤖 AI is thinking..."):
        inputs = processor(image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        output = model.generate(**inputs)
        raw_caption = processor.decode(
            output[0], skip_special_tokens=True
        )

    st.success("✅ Caption generated!")

    st.markdown("### 📝 What I See")
    st.code(raw_caption)

    with st.spinner("✨ Gemini is adding creativity..."):
        final_caption = enhance_caption(raw_caption)

    st.success("🎉 Done!")

    st.markdown("### 🌟 Final AI Caption")
    st.text_area(
        "Copy your caption:",
        final_caption,
        height=160
    )

    st.balloons()

else:
    st.info("👆 Upload an image to get started!")
