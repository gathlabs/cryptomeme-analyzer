import os
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

class MemeAnalyzer:
    def __init__(self):
        self.api_key = os.getenv('GEMINI_API_KEY')
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')

    def analyze_meme(self, image_path: str):
        """
        Analyze cryptocurrency meme and provide market prediction
        Args:
            image_path (str): Path to the meme image
        Returns:
            dict: Analysis results in JSON format
        """
        try:
            img = Image.open(image_path)
            prompt = """Analyze this cryptocurrency meme with the following conditions:
            1. Identify the main visual elements (symbols, characters, price charts)
            2. Analyze the existing text
            3. Provide a conclusion on whether the market indication is bullish or bearish
            4. Provide logical reasoning based on historical crypto patterns
            Format response IN ENGLISH and use JSON format WITHOUT MARKDOWN:
            {
                "analysis": {
                    "visual_elements": [list of elements],
                    "text_analysis": {
                        "keywords": [list of keywords],
                        "sentiment": "positive/negative/neutral"
                    }
                },
                "prediction": {
                    "trend": "bullish/bearish",
                    "confidence": 0-100,
                    "reasoning": "detailed explanation"
                }
            }"""
            
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
