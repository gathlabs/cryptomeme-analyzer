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
        "Select Page:",
        ["Meme Analyzer", "Crypto Prices", "About"]
    )

# Main content
if selected_page == "Meme Analyzer":
    st.title("🪙 Crypto Meme Based Market Predictor")
    st.markdown("""
    Market prediction analysis based on emerging memes
    """)

    uploaded_file = st.file_uploader("Upload cryptocurrency meme", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            image_path = tmp_file.name
        
        st.image(uploaded_file, caption="Uploaded Meme", use_container_width=True)
        
        if st.button("Analyze Now 🚀", type="primary"):
            # Debug for API key
            api_key = os.getenv('GEMINI_API_KEY')
            st.write(f"Debug - API Key Status: {'Set' if api_key else 'Not Set'}")
            
            # Manual input option for API key if not found
            if not api_key:
                st.warning("GEMINI_API_KEY not found in the .env file!")
                manual_api_key = st.text_input("Enter Gemini API Key manually:", type="password")
                
                if manual_api_key:
                    api_key = manual_api_key
                    os.environ['GEMINI_API_KEY'] = api_key
                else:
                    st.error("Please enter the Gemini API Key!")
                    st.stop()
            
            with st.spinner("Analyzing..."):
                try:
                    analyzer = MemeAnalyzer()
                    result = analyzer.analyze_meme(image_path)
                    
                    if "error" in result:
                        st.error(f"Error: {result['error']}")
                    else:
                        # Display analysis results
                        st.success("Analysis Complete!")
                        
                        # Market Prediction Section (Top)
                        st.markdown("## 🔮 Market Prediction")
                        
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
                        
                        # Detailed Analysis Section (Bottom, Collapsed by Default)
                        with st.expander("📊 Detailed Analysis", expanded=False):
                            # Price Analysis
                            st.subheader("💰 Price Analysis")
                            price_analysis = result.get('prediction', {}).get('price_analysis', {})
                            model_pred = price_analysis.get('model_prediction', {})
                            
                            if model_pred:
                                # Model Prediction Details
                                st.markdown("""
                                ##### Model Prediction Details
                                """)
                                
                                details_df = pd.DataFrame({
                                    'Metric': [
                                        'Current Price',
                                        'Predicted Price',
                                        'Change Percentage',
                                        'Trend',
                                        'Model Confidence',
                                        'Market Volatility',
                                        'Prediction Date'
                                    ],
                                    'Value': [
                                        f"${model_pred.get('current_price', 0):,.2f}",
                                        f"${model_pred.get('predicted_price', 0):,.2f}",
                                        f"{model_pred.get('percent_change', 0):+.2f}%",
                                        model_pred.get('trend', 'N/A').title(),
                                        f"{model_pred.get('confidence', 0):.1f}%",
                                        f"{model_pred.get('volatility', 0):.4f}",
                                        model_pred.get('prediction_date', 'N/A')
                                    ]
                                })
                                st.dataframe(details_df, hide_index=True)
                                
                                # Alignment with Meme
                                st.markdown("""
                                ##### Analysis Alignment
                                """)
                                alignment = price_analysis.get('alignment_with_meme', 'N/A')
                                if alignment.lower() == 'high':
                                    alignment_color = '#4CAF50'
                                elif alignment.lower() == 'medium':
                                    alignment_color = '#FFA500'
                                else:
                                    alignment_color = '#f44336'
                                
                                st.markdown(f"""
                                <div style='background-color: {alignment_color}; padding: 10px; border-radius: 5px; color: white;'>
                                    Alignment with Meme Analysis: {alignment.title()}
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Combined Analysis
                                st.markdown("""
                                ##### Combined Analysis
                                """)
                                st.info(price_analysis.get('combined_analysis', 'No combined analysis available'))
                            
                            # Visual Elements
                            st.markdown("""
                            ---
                            ### 🎨 Visual Elements
                            """)
                            for element in result.get('analysis', {}).get('visual_elements', []):
                                st.markdown(f"- {element}")
                            
                            # Text Analysis
                            st.markdown("""
                            ---
                            ### 📝 Text Analysis
                            """)
                            text_analysis = result.get('analysis', {}).get('text_analysis', {})
                            
                            # Sentiment with color coding
                            sentiment = text_analysis.get('sentiment', '').lower()
                            if sentiment == 'positive':
                                sentiment_color = '#4CAF50'
                            elif sentiment == 'negative':
                                sentiment_color = '#f44336'
                            else:
                                sentiment_color = '#FFA500'
                            
                            st.markdown(f"""
                            <div style='background-color: {sentiment_color}; padding: 10px; border-radius: 5px; color: white;'>
                                Sentiment: {sentiment.title()}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Keywords
                            st.markdown("#### Keywords")
                            keywords = text_analysis.get('keywords', [])
                            if keywords:
                                st.markdown(" • ".join(f"**{keyword}**" for keyword in keywords))
                
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                finally:
                    os.unlink(image_path)

elif selected_page == "Crypto Prices":
    st.header("Crypto Price Charts")
    
    # List of popular cryptocurrencies
    crypto_list = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'NEAR-USD']
    selected_crypto = st.selectbox("Select Cryptocurrency", crypto_list)
    
    try:
        # Calculate date 5 years ago from today
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.DateOffset(years=5)
        
        # Show loading state
        with st.spinner(f'Fetching data for {selected_crypto}...'):
            # Fetch price data
            crypto_data = yf.download(selected_crypto, start=start_date, end=end_date, progress=False)
            
            if crypto_data.empty:
                st.error(f"Unable to fetch data for {selected_crypto}")
            else:
                # 1. Display current price information
                latest_price = float(crypto_data['Close'].iloc[-1])
                prev_price = float(crypto_data['Close'].iloc[-2])
                price_change = ((latest_price - prev_price) / prev_price) * 100
                
                st.metric(
                    label="Current Price",
                    value=f"${latest_price:,.2f}",
                    delta=f"{price_change:.2f}%"
                )
                
                # 2. Display 7-day price table
                st.subheader("7-Day Price Table")
                last_7_days = crypto_data.tail(7).copy()
                
                # Format date and data
                formatted_data = []
                for idx, row in last_7_days.iterrows():
                    formatted_data.append({
                        'Date': idx.strftime('%Y-%m-%d'),
                        'Open': f"${float(row['Open']):,.2f}",
                        'High': f"${float(row['High']):,.2f}",
                        'Low': f"${float(row['Low']):,.2f}",
                        'Close': f"${float(row['Close']):,.2f}"
                    })
                
                # Create new DataFrame
                formatted_df = pd.DataFrame(formatted_data)
                formatted_df.set_index('Date', inplace=True)
                
                # Display table
                st.dataframe(formatted_df, use_container_width=True)
                
                # 3. Display 5-year line chart using seaborn
                st.subheader(f"{selected_crypto} Price Chart (Last 5 Years)")
                
                # Set seaborn style
                sns.set_style("whitegrid")
                plt.figure(figsize=(10, 6))
                
                # Plot data
                sns.lineplot(data=crypto_data['Close'], color='#2196f3', linewidth=2)
                
                # Configure plot
                plt.title(f'{selected_crypto} Price', pad=20)
                plt.xlabel('Date')
                plt.ylabel('Price (USD)')
                
                # Format y-axis to dollar format
                current_values = plt.gca().get_yticks()
                plt.gca().set_yticklabels(['${:,.0f}'.format(x) for x in current_values])
                
                # Rotate x-axis labels
                plt.xticks(rotation=45)
                
                # Adjust layout
                plt.tight_layout()
                
                # Display plot in Streamlit
                st.pyplot(plt)
                
                # Display additional statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    highest_price = float(crypto_data['High'].max())
                    st.metric("Highest Price", f"${highest_price:,.2f}")
                with col2:
                    lowest_price = float(crypto_data['Low'].min())
                    st.metric("Lowest Price", f"${lowest_price:,.2f}")
                with col3:
                    avg_price = float(crypto_data['Close'].mean())
                    st.metric("Average Price", f"${avg_price:,.2f}")
                
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")

else:  # About page
    st.header("🚀 CryptoMeme Analyzer")
    
    # Main Description
    st.markdown("""
    ### 🎯 Overview
    CryptoMeme Analyzer is an innovative platform that combines cryptocurrency meme analysis with AI-powered price prediction. 
    The platform leverages a powerful combination of:
    - 🤖 AI Models for visual and text meme analysis
    - 📊 LSTM Models for cryptocurrency price prediction
    - 📈 Market sentiment analysis and historical context
    """)
    
    # Features
    st.markdown("""
    ### ✨ Key Features
    
    #### 1. Comprehensive Meme Analysis
    - **Visual Analysis**: Detection of visual elements, symbols, characters, and charts
    - **Text Analysis**: In-depth analysis of text and context within memes
    - **Sentiment Analysis**: Market sentiment determination (bullish/bearish)
    
    #### 2. AI Price Prediction
    - **Real-time Price Data**: Up-to-the-minute cryptocurrency pricing
    - **LSTM Predictions**: Deep learning-based price forecasting
    - **Volatility Analysis**: Market volatility calculations for confidence metrics
    
    #### 3. Market Context
    - **Historical Patterns**: Analysis of historical price patterns
    - **Combined Analysis**: Integration of meme analysis with price predictions
    - **Confidence Metrics**: Trust scores based on multiple factors
    """)
    
    # How It Works
    with st.expander("🔍 How It Works", expanded=False):
        st.markdown("""
        1. **Upload Meme**
           - Upload your cryptocurrency meme for analysis
           - Supported formats: JPG, JPEG, PNG
        
        2. **AI Analysis**
           - AI models analyze visual elements and text
           - Identify key symbols and context
           - Determine base meme sentiment
        
        3. **Price Analysis**
           - LSTM model analyzes historical price data
           - Calculates price predictions and volatility
           - Generates confidence metrics
        
        4. **Combined Prediction**
           - Merges meme analysis and price predictions
           - Produces final prediction with confidence score
           - Provides market trend recommendations
        """)
    
    # Technical Details
    with st.expander("⚙️ Technical Details", expanded=False):
        st.markdown("""
        #### Models & Technology
        - **Vision AI**: Google Cloud Vision AI for image analysis
        - **NLP**: Gemini Pro for text and context analysis
        - **Price Prediction**: Custom LSTM model built with PyTorch
        - **Data Source**: Real-time data from Yahoo Finance
        
        #### Metrics & Calculations
        - **Volatility**: Standard deviation of relative price changes
        - **Confidence Score**: Combination of market volatility and prediction quality
        - **Price Change**: Percentage change from current to predicted price
        """)
    
    # Disclaimer
    st.warning("""
    **⚠️ Disclaimer**
    
    This application is intended for educational and entertainment purposes only. The analysis and predictions 
    provided should NOT be used as the sole basis for investment decisions. Always conduct your own research 
    and consult with professional financial advisors before making any investment decisions.
    """)
    
    # Version Info
    st.sidebar.markdown("---")
    st.sidebar.markdown("v1.1.0")
    st.sidebar.markdown("Made with ❤️ by CryptoMeme Team")
