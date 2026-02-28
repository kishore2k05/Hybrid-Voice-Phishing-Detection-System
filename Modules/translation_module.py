import os
import gemini_module

INPUT_FOLDER = '../datasets/raw_transcripts'
OUTPUT_FOLDER = '../datasets/translated_transcripts'

def translate_all_files():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    if not os.path.exists(INPUT_FOLDER):
        print(f"Error: Input folder '{INPUT_FOLDER}' missing.")
        return

    files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".txt")]
    print(f"🚀 Processing {len(files)} files using Google Gemini...\n")

    for filename in files:
        file_path = os.path.join(INPUT_FOLDER, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            english_text = f.read()

        print(f"{filename}...")
        tanglish = gemini_module.convert_with_gemini(english_text, "Tanglish")
        
        if tanglish:
            new_name = filename.replace(".txt", "_tanglish.txt")
            with open(os.path.join(OUTPUT_FOLDER, new_name), "w", encoding="utf-8") as f:
                f.write(tanglish)
            print(f"Tanglish Saved")

        hinglish = gemini_module.convert_with_gemini(english_text, "Hinglish")
        
        if hinglish:
            new_name = filename.replace(".txt", "_hinglish.txt")
            with open(os.path.join(OUTPUT_FOLDER, new_name), "w", encoding="utf-8") as f:
                f.write(hinglish)
            print(f"Hinglish Saved")

    print("\n🎉 Translation Complete!")

if __name__ == "__main__":
    translate_all_files()