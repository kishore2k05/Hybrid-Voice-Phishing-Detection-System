use std::env;
use std::fs;

fn main() {
    println!("Text Cleaner Started");
    
    let args: Vec<String> = env::args().collect();
    
    let folder_type = if args.len() > 2 {
        &args[2]
    } else {
        "raw"
    };
    
    let filename = if args.len() > 1 {
        args[1].clone()
    } else {
        String::from("normal_01")
    };
    
    let input_folder = match folder_type {
        "translated" => "../datasets/translated_transcripts",
        _ => "../datasets/raw_transcripts"
    };
    
    let input_path = format!("{}/{}.txt", input_folder, filename);
    let output_path = format!("../datasets/cleaned_transcripts/{}.txt", filename);
    
    let _ = fs::create_dir_all("../datasets/cleaned_transcripts");
    
    println!("Reading: {}", input_path);
    
    let transcript = match fs::read_to_string(&input_path) {
        Ok(content) => {
            println!("Successfully read transcript");
            content
        }
        Err(e) => {
            eprintln!("Error reading file: {}", e);
            return;
        }
    };
    
    println!("Original: {} characters", transcript.len());
    
    let cleaned_text = clean_text(&transcript);
    
    println!("Cleaned: {} characters", cleaned_text.len());
    
    match fs::write(&output_path, &cleaned_text) {
        Ok(_) => println!("Saved to: {}", output_path),
        Err(e) => eprintln!("Error saving: {}", e),
    }
}

fn clean_text(text: &str) -> String {
    println!("Cleaning text");
    
    let mut cleaned = String::new();
    let lowercase = text.to_lowercase();
    
    for character in lowercase.chars() {
        if character.is_alphabetic() || character.is_whitespace() {
            cleaned.push(character);
        }
    }
    
    let words: Vec<&str> = cleaned
        .split_whitespace()
        .filter(|word| !word.is_empty())
        .collect();
    
    println!("Word count: {}", words.len());
    
    words.join(" ")
}