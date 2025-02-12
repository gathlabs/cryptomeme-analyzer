import os
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

class MemeAnalyzer:
    def __init__(self):
        self.api_key = os.getenv('GEMINI_API_KEY')
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def analyze_meme(self, image_path: str):
        """
        Menganalisis meme cryptocurrency dan memberikan prediksi market
        Args:
            image_path (str): Path ke gambar meme
        Returns:
            dict: Hasil analisis dalam format JSON
        """
        try:
            img = Image.open(image_path)
            prompt = """Analisis meme cryptocurrency ini dengan ketentuan:
            1. Identifikasi elemen visual utama (simbol, karakter, grafik harga)
            2. Analisis teks yang ada
            3. Berikan kesimpulan apakah indikasi market akan bullish atau bearish
            4. Berikan reasoning logis berdasarkan pola historis crypto
            Format respons DALAM BAHASA INDONESIA dan gunakan format JSON TANPA MARKDOWN:
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
            # Bersihkan response dari karakter non-JSON
            cleaned = response_text.strip()
            if '```json' in cleaned:
                cleaned = cleaned.split('```json')[1].split('```')[0]
            elif '```' in cleaned:
                cleaned = cleaned.split('```')[1]
            
            # Parse JSON
            response_json = json.loads(cleaned)
            
            # Validasi struktur response
            if not all(key in response_json for key in ['analysis', 'prediction']):
                raise ValueError("Struktur response tidak valid")
                
            return response_json
        except Exception as e:
            return {
                "error": f"Gagal memparse respons Gemini: {str(e)}",
                "raw_response": response_text
            }

if __name__ == "__main__":
    analyzer = MemeAnalyzer()
    result = analyzer.analyze_meme("path/to/meme.jpg")
    print(result)
