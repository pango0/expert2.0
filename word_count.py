import os
import PyPDF2
from collections import Counter

def count_characters_in_chinese_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    text = ''.join(text.split())
    return Counter(text)

def process_pdf_directory(directory_path):
    total_char_counts = Counter()
    
    for filename in os.listdir(directory_path):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(directory_path, filename)
            print(f"Processing: {filename}")
            char_counts = count_characters_in_chinese_pdf(pdf_path)
            total_char_counts += char_counts
    
    return total_char_counts

# Example usage
directory_path = 'data'
char_counts = process_pdf_directory(directory_path)

total_chars = sum(char_counts.values())
print(f"\nTotal number of characters: {total_chars}")