import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# ===== DICTIONARY PENJELASAN MANFAAT (Tetap) =====
insights = {
    "hijau": "Pisang dengan tingkat kematangan **Hijau** baik untuk mengontrol gula darah karena tinggi pati resisten dan menjaga kesehatan pencernaan.",
    "kuning": "Pisang dengan tingkat kematangan **Kuning** bermanfaat sebagai sumber energi yang ideal, kaya potasium untuk kesehatan jantung dan serat untuk pencernaan.",
    "kuning bintik cokelat": "Pisang dengan tingkat kematangan **Kuning bintik cokelat** sangat mudah dicerna, kaya antioksidan, dan memiliki kandungan yang baik untuk imunitas.",
    "cokelat": "Pisang dengan tingkat kematangan **cokelat** memiliki kandungan gula dan antioksidan sangat tinggi, sempurna untuk pemanis alami pada makanan."
}

detailed_benefits = {
    "hijau": [
        "**Tinggi Pati Resisten**: Pati ini berfungsi seperti serat, tidak cepat diubah menjadi gula, sehingga membantu menjaga kadar gula darah tetap stabil.",
        "**Menjaga Kesehatan Usus**: Merupakan sumber prebiotik yang menjadi makanan bagi bakteri baik di usus Anda.",
        "**Memberi Rasa Kenyang Lebih Lama**: Kandungan patinya yang kompleks membuat Anda merasa kenyang lebih lama, cocok untuk program diet.",
    ],
    "kuning": [
        "**Kaya Potasium**: Penting untuk mengatur tekanan darah, fungsi saraf, dan keseimbangan cairan dalam tubuh.",
        "**Sumber Serat Pektin**: Membantu melancarkan pencernaan dan dapat mencegah sembelit.",
        "**Kaya Antioksidan**: Mengandung dopamin dan katekin yang melindungi tubuh dari kerusakan sel.",
        "**Mudah Dicerna**: Gula alami dalam pisang kuning lebih mudah dipecah oleh tubuh untuk menjadi energi cepat."
    ],
    "kuning bintik cokelat": [
        "**Puncak Antioksidan**: Bintik cokelat menandakan bahwa pati telah berubah menjadi gula sederhana dan tingkat antioksidan berada di puncaknya.",
        "**Mengandung TNF (Tumor Necrosis Factor)**: Kandungan TNF-nya dapat membantu melawan sel-sel abnormal dalam tubuh.",
        "**Sangat Mudah Dicerna**: Cocok untuk orang dengan sistem pencernaan sensitif karena patinya sudah terurai.",
        "**Rasa Paling Manis**: Ideal sebagai pemanis alami dalam smoothie atau kue."
    ],
    "cokelat": [
        "**Kaya Triptofan**: Asam amino ini diubah tubuh menjadi serotonin, yang membantu meningkatkan mood dan membuat rileks.",
        "**Sumber Antioksidan Maksimal**: Proses pematangan lanjut meningkatkan konsentrasi antioksidan.",
        "**Pemanis Alami Terbaik**: Hampir seluruh patinya telah menjadi gula, membuatnya sangat ideal untuk bahan kue pisang atau pancake tanpa tambahan gula."
    ]
}

# ===== MUAT MODEL (.h5) =====
try:
    model = tf.keras.models.load_model("model_mobilenetv2_2.h5")
except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")
    st.stop()

# ===== LABEL MANUAL SESUAI URUTAN TRAINING =====
labels = ['cokelat', 'hijau', 'kuning', 'kuning bintik cokelat', 'unknown']

# ===== FUNGSI PROSES GAMBAR =====
def crop_image(image):
    crop_size = min(image.width, image.height) * 0.6
    crop_x = (image.width - crop_size) / 2
    crop_y = (image.height - crop_size) / 2
    return image.crop((crop_x, crop_y, crop_x + crop_size, crop_y + crop_size))

def classify_image(image):
    image = image.convert("RGB")

    # Crop dan resize
    cropped_image = crop_image(image)
    img = cropped_image.resize((150, 150))

    # Normalisasi dan ubah jadi batch
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1, 150, 150, 3)

    # Klasifikasi
    prediction = model.predict(img_array)
    max_index = np.argmax(prediction)
    return labels[max_index], prediction[0][max_index]


# ===== SIDEBAR =====
with st.sidebar:
    st.image("pisang.png", use_column_width=True)
    st.header("🍌 Klasifikasi Tingkat Kematangan Pisang 🍌")
    menu = st.radio("Pilih Menu", ["Detail Manfaat Pisang", "Klasifikasi Pisang"])

# ===== HALAMAN KLASIFIKASI =====
if menu == "Klasifikasi Pisang":
    st.title("📷 Unggah Gambar Pisang")
    
    st.info("""
    📝 **Tips:**
    - Pastikan gambar pisang yang diunggah **jelas** dan memiliki **pencahayaan yang baik**.
    - Gunakan **gambar dari sudut yang tepat** agar bentuk dan warna terlihat.
    - Jika hasil tidak sesuai, kemungkinan disebabkan oleh **kualitas gambar yang kurang bagus**.
    """)

    
    uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang diunggah", use_column_width=True)
        st.write("---")
        with st.spinner("Sedang melakukan klasifikasi...🔍"):
            label, prob = classify_image(image)

        if label == "unknown":
            st.warning("⚠️ Gambar yang diunggah tidak dikenali sebagai pisang. Silakan unggah gambar pisang yang jelas.")
        else:
            st.success(f"✅ Hasil: {label.capitalize()} ({prob:.2%})")
            st.info(f"💡 Insight: {insights.get(label, '-')}")
            st.markdown("ℹ️ Lihat penjelasan lengkap di menu **Detail Manfaat Pisang**.")

# ===== HALAMAN DETAIL MANFAAT =====
elif menu == "Detail Manfaat Pisang":
    st.title("🩺 Manfaat Konsumsi Pisang Berdasarkan Warna 🍌")
    st.write("Setiap warna pisang menunjukkan kandungan nutrisi yang berbeda.")
    st.divider()
    for warna, detail in detailed_benefits.items():
        st.subheader(f"Pisang {warna.capitalize()}")
        for point in detail:
            st.markdown(f"✅ {point}")
        st.divider()
