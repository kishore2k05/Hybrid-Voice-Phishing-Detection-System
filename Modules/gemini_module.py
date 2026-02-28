from google import genai
from google.genai import types
import time
from dotenv import load_dotenv
import os

load_dotenv()
GEMINI_KEYS = [
    os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY_2"),
    os.getenv("GEMINI_API_KEY_3"),
    os.getenv("GEMINI_API_KEY_4"),
    os.getenv("GEMINI_API_KEY_5")
]

GEMINI_KEYS = [key for key in GEMINI_KEYS if key]

if not GEMINI_KEYS:
    raise ValueError("No API keys found! Make sure you created a .env file with GEMINI_API_KEY_1")

MODEL_NAME = "gemini-2.0-flash" 

current_key_index = 0

def convert_with_gemini(text, target_language):
    global current_key_index
    
    TANGLISH_PROMPT = """
You are a native Tamil speaker from Chennai. Translate the following English dialogue into 'Tanglish' (Tamil words written in English script).
RULES:
1. You MUST use Tamil verbs for actions (e.g., 'work aagala', 'restart pannunga').
2. Keep technical nouns in English (Printer, Network, Wi-Fi, Bank, OTP).
3. Do NOT use Hindi words.
4. Output ONLY the translated text string.
"""

    HINGLISH_PROMPT = """
You are a native Hindi speaker. Translate the following English dialogue into 'Hinglish'.
RULES:
1. Use natural Hindi grammar (e.g., 'kaam nahi kar raha hai').
2. Keep technical nouns in English.
3. Output ONLY the translated text string.
"""
    max_retries = 3

    for attempt in range(max_retries):
        try:
            api_key = GEMINI_KEYS[current_key_index]
            client = genai.Client(api_key=api_key)
            
            if target_language == "Tanglish":
                full_prompt = TANGLISH_PROMPT + f"\n\nTEXT TO TRANSLATE:\n{text}"
            else:
                full_prompt = HINGLISH_PROMPT + f"\n\nTEXT TO TRANSLATE:\n{text}"
            
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3, 
                )
            )
            
            if response.text:
                return response.text.strip()
            
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "ResourceExhausted" in error_msg:
                print(f"Key #{current_key_index + 1} quota exceeded. Switching...")
                current_key_index = (current_key_index + 1) % len(GEMINI_KEYS)
                time.sleep(1)
            else:
                print(f"Error: {e}")
                return None
    
    return None