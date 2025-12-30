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
2️⃣ AI describes what it sees  
3️⃣ Gemini makes it creative ✨  
4️⃣ Copy & share 🎉  
""")

st.sidebar.markdown("---")
st.sidebar.info("⚠️ AI-generated content. For creative use only.")

# ---------------- MAIN TITLE ----------------
st.markdown(
    "<h1 style='text-align: center;'>🖼️ AI Image Caption Generator</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center;'>From image → meaning → creativity ✨</p>",
    unsafe_allow_html=True
)

st.markdown("---")

# ---------------- GEMINI SETUP (STABLE) ----------------
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=GEMINI_API_KEY)

gemini_model = genai.GenerativeModel("gemini-pro")

# ---------------- LOAD BLIP MODEL ----------------
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

# ---------------- GEMINI ENHANCEMENT FUNCTION ----------------
def enhance_caption(raw_caption):
    prompt = f"""
You are a creative social media assistant.

STRICT RULES:
- Do NOT repeat the same caption every time
- Do NOT reuse the same hashtags
- Caption must feel unique and natural
- Avoid generic phrases like "A moment captured beautifully"

TASK:
1. "What I see": one factual sentence.
2. "Caption for You": creative caption with emojis and 2–4 relevant hashtags.

Image description:
{raw_caption}

FORMAT EXACTLY AS:

What I see:
<one sentence>

Caption for You:
<creative caption>
"""

    response = gemini_model.generate_content(prompt)

    if not response or not response.text:
        return f"""What I see:
{raw_caption}

Caption for You:
A unique moment captured through AI ✨
"""

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
    st.image(image, width=500)

    with st.spinner("🤖 Understanding the image..."):
        inputs = processor(image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        output = model.generate(**inputs)
        raw_caption = processor.decode(
            output[0], skip_special_tokens=True
        )

    st.success("✅ Image understood")

    # 🔹 RAW CAPTION BACK (CLEARLY SHOWN)
    st.markdown("### 🔍 Raw Caption")
    st.code(raw_caption)

    with st.spinner("✨ Making it creative with Gemini..."):
        final_caption = enhance_caption(raw_caption)

    st.success("🎉 Final Caption Ready")

    st.markdown("### 🌟 Final AI Caption")
    st.text_area(
        "Copy your caption:",
        final_caption,
        height=180
    )

    st.balloons()

else:
    st.info("👆 Upload an image to get started")
