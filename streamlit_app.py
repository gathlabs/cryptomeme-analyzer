import streamlit as st
from app.analyzer import MemeAnalyzer
import tempfile
import os
import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

st.set_page_config(
    page_title="Crypto Meme Analyzer",
    page_icon="🚀",
    layout="centered"
)

# Sidebar configuration
with st.sidebar:
    st.title("Menu")
    selected_page = st.radio(
        "Pilih Halaman:",
        ["Meme Analyzer", "Crypto Prices", "About"]
    )

# Main content
if selected_page == "Meme Analyzer":
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
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                st.error("GEMINI_API_KEY tidak ditemukan di file .env!")
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
                                        <h2>📈 Bullish (Confidence: {confidence}%)</h2>
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

elif selected_page == "Crypto Prices":
    st.header("Crypto Price Charts")
    
    # Daftar kripto populer
    crypto_list = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD', 'DOGE-USD']
    selected_crypto = st.selectbox("Pilih Cryptocurrency", crypto_list)
    
    try:
        # Hitung tanggal 5 tahun yang lalu dari hari ini
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.DateOffset(years=5)
        
        # Tampilkan loading state
        with st.spinner(f'Mengambil data {selected_crypto}...'):
            # Ambil data harga
            crypto_data = yf.download(selected_crypto, start=start_date, end=end_date, progress=False)
            
            if crypto_data.empty:
                st.error(f"Tidak dapat mengambil data untuk {selected_crypto}")
            else:
                # 1. Tampilkan informasi harga terkini
                latest_price = float(crypto_data['Close'].iloc[-1])
                prev_price = float(crypto_data['Close'].iloc[-2])
                price_change = ((latest_price - prev_price) / prev_price) * 100
                
                st.metric(
                    label="Harga Terkini",
                    value=f"${latest_price:,.2f}",
                    delta=f"{price_change:.2f}%"
                )
                
                # 2. Tampilkan tabel harga 7 hari terakhir
                st.subheader("Harga 7 Hari Terakhir")
                last_7_days = crypto_data.tail(7).copy()
                
                # Format tanggal dan data
                formatted_data = []
                for idx, row in last_7_days.iterrows():
                    formatted_data.append({
                        'Tanggal': idx.strftime('%Y-%m-%d'),
                        'Open': f"${float(row['Open']):,.2f}",
                        'High': f"${float(row['High']):,.2f}",
                        'Low': f"${float(row['Low']):,.2f}",
                        'Close': f"${float(row['Close']):,.2f}"
                    })
                
                # Buat DataFrame baru
                formatted_df = pd.DataFrame(formatted_data)
                formatted_df.set_index('Tanggal', inplace=True)
                
                # Tampilkan tabel
                st.dataframe(formatted_df, use_container_width=True)
                
                # 3. Tampilkan line chart 5 tahun terakhir menggunakan seaborn
                st.subheader(f"Grafik Harga {selected_crypto} (5 Tahun Terakhir)")
                
                # Set style seaborn
                sns.set_style("whitegrid")
                plt.figure(figsize=(10, 6))
                
                # Plot data
                sns.lineplot(data=crypto_data['Close'], color='#2196f3', linewidth=2)
                
                # Konfigurasi plot
                plt.title(f'Harga {selected_crypto}', pad=20)
                plt.xlabel('Tanggal')
                plt.ylabel('Harga (USD)')
                
                # Format y-axis ke format dollar
                current_values = plt.gca().get_yticks()
                plt.gca().set_yticklabels(['${:,.0f}'.format(x) for x in current_values])
                
                # Rotate x-axis labels
                plt.xticks(rotation=45)
                
                # Adjust layout
                plt.tight_layout()
                
                # Tampilkan plot di Streamlit
                st.pyplot(plt)
                
                # Tampilkan statistik tambahan
                col1, col2, col3 = st.columns(3)
                with col1:
                    highest_price = float(crypto_data['High'].max())
                    st.metric("Harga Tertinggi", f"${highest_price:,.2f}")
                with col2:
                    lowest_price = float(crypto_data['Low'].min())
                    st.metric("Harga Terendah", f"${lowest_price:,.2f}")
                with col3:
                    avg_price = float(crypto_data['Close'].mean())
                    st.metric("Harga Rata-rata", f"${avg_price:,.2f}")
                
    except Exception as e:
        st.error(f"Error saat mengambil data: {str(e)}")

else:  # About page
    st.header("🚀 About Crypto Meme Analyzer")
    
    # Deskripsi Utama
    st.markdown("""
    ### 🎯 Apa itu Crypto Meme Analyzer?
    
    Crypto Meme Analyzer adalah aplikasi inovatif yang menggabungkan kekuatan AI dengan analisis pasar cryptocurrency. 
    Aplikasi ini memungkinkan Anda untuk:
    - 🔍 Menganalisis meme crypto menggunakan Google Gemini AI
    - 📊 Memantau harga real-time cryptocurrency populer
    - 📈 Melihat tren harga historis 5 tahun terakhir
    
    ### 🛠️ Fitur Utama
    
    #### 1. Meme Analyzer
    - Analisis visual dan teks dari meme cryptocurrency
    - Prediksi sentimen market (Bullish/Bearish)
    - Penjelasan detail tentang elemen visual dan konteks
    - Tingkat kepercayaan prediksi
    
    #### 2. Crypto Price Tracker
    - Data harga real-time dari Yahoo Finance
    - Visualisasi harga 5 tahun terakhir
    - Statistik harga (tertinggi, terendah, rata-rata)
    - Tabel harga 7 hari terakhir
    
    ### 🔧 Teknologi yang Digunakan
    
    Aplikasi ini dibangun menggunakan teknologi modern:
    - **Streamlit**: Framework Python untuk aplikasi web
    - **Google Gemini AI**: Model AI untuk analisis meme
    - **Yahoo Finance API**: Sumber data harga crypto
    - **Seaborn & Matplotlib**: Visualisasi data
    - **Pandas**: Analisis dan manipulasi data
    
    ### 📝 Cara Penggunaan
    
    1. **Analisis Meme**:
       - Upload gambar meme cryptocurrency
       - Klik "Analisis Sekarang"
       - Dapatkan prediksi dan analisis detail
    
    2. **Track Harga**:
       - Pilih cryptocurrency dari daftar
       - Lihat harga terkini dan statistik
       - Analisis tren harga melalui grafik
    
    ### ⚠️ Disclaimer
    
    Aplikasi ini dibuat untuk tujuan edukasi dan hiburan. Prediksi dan analisis yang diberikan tidak boleh dianggap sebagai 
    saran finansial. Selalu lakukan riset mandiri sebelum membuat keputusan investasi.
    
    ### 👨‍💻 Developer
    
    Dikembangkan dengan ❤️ oleh Tim Gathlabs
    """)
    
    # Tambahkan statistik aplikasi dalam 3 kolom
    st.markdown("### 📊 Statistik Aplikasi")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Cryptocurrency", value="5+", help="Jumlah cryptocurrency yang dapat dianalisis")
    with col2:
        st.metric(label="Data Historis", value="5 Tahun", help="Rentang data historis yang tersedia")
    with col3:
        st.metric(label="Update Harga", value="Real-time", help="Frekuensi update data harga")
