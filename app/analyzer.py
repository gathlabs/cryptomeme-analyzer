import os
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv
from .price_predictor import get_price_prediction

load_dotenv()

class MemeAnalyzer:
    def __init__(self):
        self.api_key = os.getenv('GEMINI_API_KEY')
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')

    def analyze_meme(self, image_path: str, symbol: str = 'BTC-USD'):
        """
        Analyze cryptocurrency meme and provide market prediction
        Args:
            image_path (str): Path to the meme image
            symbol (str): Cryptocurrency symbol
        Returns:
            dict: Analysis results in JSON format
        """
        try:
            img = Image.open(image_path)
            
            # Get price prediction
            try:
                price_pred = get_price_prediction(symbol)
                price_context = f"""
                Consider the following price prediction data:
                - Current price: ${price_pred['current_price']:.2f}
                - Predicted next price: ${price_pred['predicted_price']:.2f}
                - Predicted price change: {price_pred['percent_change']:+.2f}%
                - Model prediction trend: {price_pred['trend']}
                - Model confidence: {price_pred['confidence']:.1f}%
                - Market volatility: {price_pred['volatility']:.4f}
                - Prediction for: {price_pred['prediction_date']}
                """
            except Exception as e:
                print(f"Warning: Price prediction failed: {str(e)}")
                price_context = ""
                price_pred = None
            
            prompt = f"""Analyze this cryptocurrency meme with the following conditions:
            1. Identify the main visual elements (symbols, characters, price charts)
            2. Analyze the existing text
            3. Provide a conclusion on whether the market indication is bullish or bearish
            4. Provide logical reasoning based on historical crypto patterns
            {price_context}
            Format response IN ENGLISH and use JSON format WITHOUT MARKDOWN:
            {{
                "analysis": {{
                    "visual_elements": [list of elements],
                    "text_analysis": {{
                        "keywords": [list of keywords],
                        "sentiment": "positive/negative/neutral"
                    }}
                }},
                "prediction": {{
                    "trend": "bullish/bearish",
                    "confidence": 0-100,
                    "reasoning": "detailed explanation",
                    "price_analysis": {{
                        "model_prediction": {{ "null" if price_pred is None else {{
                            "current_price": {price_pred['current_price']:.2f},
                            "predicted_price": {price_pred['predicted_price']:.2f},
                            "percent_change": {price_pred['percent_change']:.2f},
                            "trend": "{price_pred['trend']}",
                            "confidence": {price_pred['confidence']:.1f},
                            "volatility": {price_pred['volatility']:.4f},
                            "prediction_date": "{price_pred['prediction_date']}"
                        }} }},
                        "alignment_with_meme": "high/medium/low",
                        "combined_analysis": "explanation of how the price prediction aligns with meme analysis"
                    }}
                }}
            }}"""
            
            response = self.model.generate_content([prompt, img])
            return self._parse_response(response.text)
        except Exception as e:
            return {"error": str(e)}

    def _parse_response(self, response_text: str):
        try:
            import json
            # Clean the response from non-JSON characters
            cleaned = response_text.strip()
            if '```json' in cleaned:
                cleaned = cleaned.split('```json')[1].split('```')[0]
            elif '```' in cleaned:
                cleaned = cleaned.split('```')[1]
            
            # Parse JSON
            response_json = json.loads(cleaned)
            
            # Validate the response structure
            if not all(key in response_json for key in ['analysis', 'prediction']):
                raise ValueError("Invalid response structure")
                
            return response_json
        except Exception as e:
            return {
                "error": f"Failed to parse Gemini response: {str(e)}",
                "raw_response": response_text
            }

if __name__ == "__main__":
    analyzer = MemeAnalyzer()
    result = analyzer.analyze_meme("path/to/meme.jpg")
    print(result)
