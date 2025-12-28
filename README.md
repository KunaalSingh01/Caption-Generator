# 🖼️ Caption Generator

Caption Generator is a web-based AI application that generates meaningful and creative captions for images.  
The system first understands the image using a pre-trained image captioning model and then enhances the caption using **Google Gemini AI** to make it more engaging and user-friendly.

The application is built using **Streamlit** and can be deployed easily on **Streamlit Cloud**.

---

## 🚀 Features

- Upload an image (JPG / PNG / JPEG)
- Generate a clear image description (“What I see”)
- Generate a creative caption for the user
- Fun and interactive Streamlit UI
- Real-time caption generation
- Secure handling of Gemini API key

---

## 🛠️ Technologies Used

- Python  
- Streamlit  
- BLIP Image Captioning Model (Salesforce)  
- Google Gemini API  
- Hugging Face Transformers  
- PyTorch  
- Pillow  
- Git & GitHub  
- Streamlit Cloud  

---

## 📁 Project Structure

caption-generator/
│
├── streamlit_app.py # Main Streamlit application
├── requirements.txt # Required Python libraries
├── README.md # Project documentation
├── .gitignore # Ignored files
│
├── .streamlit/
│ └── secrets.toml # Gemini API key (not pushed to GitHub)
│
└── venv/ # Virtual environment (ignored)
