import streamlit as st
import json
import base64
from ultralytics import YOLO
from openai import OpenAI
import os
from PIL import Image

# --- AYARLAR VE YAPILANDIRMA ---
# ÖNEMLİ: Canlıda API anahtarını koda yazmak yerine Streamlit "Secrets" kullanacağız.
API_KEY = st.secrets.get("OPENAI_API_KEY", "")
MODEL_YOLU = "best.pt" 

st.set_page_config(page_title="Askeri İstihbarat Analizi", page_icon="🛡️", layout="wide")

@st.cache_resource
def model_yukle():
    return YOLO(MODEL_YOLU)

model = model_yukle()
client = OpenAI(api_key=API_KEY)

# --- YAN MENÜ (AYARLAR) ---
st.sidebar.header("⚙️ Sistem Ayarları")
# Kullanıcının isteği üzerine varsayılan %40 (0.40) yapıldı
conf_val = st.sidebar.slider("Güven Eşiği (Confidence)", 0.0, 1.0, 0.40)

# --- FONKSİYONLAR ---
def analiz_et(image_path, threshold):
    results = model(image_path, verbose=False)
    filtrelenmis_tespitler = []
    
    for result in results:
        for box in result.boxes:
            guven_skoru = float(box.conf[0])
            if guven_skoru >= threshold:
                sinif_id = int(box.cls[0])
                sinif_adi = model.names[sinif_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                filtrelenmis_tespitler.append({
                    "unsur": sinif_adi,
                    "guven_skoru_yuzde": int(guven_skoru * 100),
                    "koordinatlar": [x1, y1, x2, y2]
                })
    return filtrelenmis_tespitler

def gpt4o_rapor_olustur(image_path, tespit_verisi):
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
    system_prompt = """Sen kıdemli bir askeri istihbarat analistisin. Görseli ve tespit verilerini analiz et. 
    Türkçe, profesyonel ve taktiksel bir rapor sun."""

    user_content = [
        {"type": "text", "text": f"Tespitler: {json.dumps(tespit_verisi)}"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}],
        max_tokens=800
    )
    return response.choices[0].message.content

# --- ANA ARAYÜZ ---
st.title("🛡️ Otonom Askeri İstihbarat Sistemi")
yuklenen_dosya = st.file_uploader("Görsel Yükle", type=["jpg", "jpeg", "png"])

if yuklenen_dosya:
    temp_path = "temp_upload.jpg"
    with open(temp_path, "wb") as f:
        f.write(yuklenen_dosya.getbuffer())
        
    c1, c2 = st.columns(2)
    with c1:
        st.image(Image.open(temp_path), caption="Yüklenen Görsel", use_column_width=True)
        
    with c2:
        with st.spinner("Analiz ediliyor..."):
            sonuclar = analiz_et(temp_path, conf_val)
            if sonuclar:
                rapor = gpt4o_rapor_olustur(temp_path, sonuclar)
                st.info(f"Tespit Edilen Unsur Sayısı: {len(sonuclar)}")
                st.markdown(rapor)
            else:
                st.warning(f"%{int(conf_val*100)} üzerinde unsur bulunamadı.")