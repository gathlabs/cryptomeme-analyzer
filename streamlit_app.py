import streamlit as st
from app.analyzer import MemeAnalyzer
import tempfile
import os

st.set_page_config(
    page_title="Crypto Meme Analyzer",
    page_icon="🚀",
    layout="centered"
)

# Sidebar configuration
with st.sidebar:
    st.title("Pengaturan")
    api_key = st.text_input("Masukkan Gemini API Key", type="password")
    os.environ['GEMINI_API_KEY'] = api_key

# Main interface
st.title("🪙 Crypto Meme Based Market Predictor")
st.markdown("""
Analisis prediksi market berdasarkan meme yang muncul
""")

uploaded_file = st.file_uploader("Unggah meme cryptocurrency", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        image_path = tmp_file.name
    
    st.image(uploaded_file, caption="Meme yang Diunggah", use_container_width=True)
    
    if st.button("Analisis Sekarang 🚀", type="primary"):
        if not api_key:
            st.error("Harap masukkan API Key Gemini di sidebar!")
            st.stop()
        
        with st.spinner("Analyzing..."):
            try:
                analyzer = MemeAnalyzer()
                result = analyzer.analyze_meme(image_path)
                
                if "error" in result:
                    st.error(f"Error: {result['error']}")
                else:
                    # Tampilkan hasil analisis
                    st.success("Analisis Selesai!")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        with st.expander("📊 Detail Analisis", expanded=True):
                            st.subheader("Elemen Visual")
                            for element in result.get('analysis', {}).get('visual_elements', []):
                                st.markdown(f"- {element}")
                            
                            st.subheader("Analisis Teks")
                            st.write(f"Sentimen: **{result.get('analysis', {}).get('text_analysis', {}).get('sentiment', '')}**")
                            st.write("Kata Kunci:")
                            st.write(result.get('analysis', {}).get('text_analysis', {}).get('keywords', []))
                    
                    with col2:
                        with st.expander("🔮 Prediksi Market", expanded=True):
                            trend = result.get('prediction', {}).get('trend', '')
                            confidence = result.get('prediction', {}).get('confidence', 0)
                            
                            if trend.lower() == "bullish":
                                st.markdown(f"""
                                <div style='background-color: #4CAF50; padding: 20px; border-radius: 10px; color: white;'>
                                    <h2>📈 Bullish ({confidence}%)</h2>
                                    <p>{result.get('prediction', {}).get('reasoning', '')}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div style='background-color: #f44336; padding: 20px; border-radius: 10px; color: white;'>
                                    <h2>📉 Bearish ({confidence}%)</h2>
                                    <p>{result.get('prediction', {}).get('reasoning', '')}</p>
                                </div>
                                """, unsafe_allow_html=True)
            
            except Exception as e:
                st.error(f"Terjadi error: {str(e)}")
            finally:
                os.unlink(image_path)
