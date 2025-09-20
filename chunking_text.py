import nltk
nltk.download('punkt')
nltk.download('punkt_tab')  # new in recent NLTK versions
from nltk.tokenize import sent_tokenize
import json
import os


def save_chunks_to_json(chunks, file_name):
    base_path = os.getcwd()
    file_name = base_path + f'/pdf-store/{file_name}'
    print(f"Total chunks: {len(chunks)}")
    chunk_data = [{"chunk_id": i, "chunk_value": chunk} for i, chunk in enumerate(chunks, start=1)]
    # Save JSON file
    output_file = str(file_name.split(".")[0])+ '.json'
    with open(output_file, "w") as f:
        json.dump(chunk_data, f, indent=4)

def sentence_chunk(file_name, max_tokens=100):
    with open(file_name, 'r') as f:
        text = f.read()
    sentences = sent_tokenize(text)
    chunks, current = [], []
    length = 0
    for sent in sentences:
        length += len(sent.split())
        if length > max_tokens:
            chunks.append(" ".join(current))
            current, length = [], 0
        current.append(sent)
    if current:
        chunks.append(" ".join(current))
    save_chunks_to_json(chunks, file_name)



if __name__ == "__main__":
    filename = 'Mahindra_Thar_Car_Manual.txt'



