import json
import os 

prompts = [
            "Alice lives in France, Paris - Alice, John lives in Germany, Berlin - John, Peter lives in USA, Washington - Peter",
            "Lucy lives in Turkey, Ankara - Lucy, Sara lives in Italy, Rome - Sara, Bob lives in Spain, Madrid - Bob",
            "Tom lives in Canada, Toronto - Tom, Anna lives in Australia, Canberra - Anna, Michael lives in Japan, Tokyo - Michael",
            "David lives in Brazil, Rio de Janeiro - David, Alice lives in France, Paris - Alice, Peter lives in Germany, Berlin - Peter",
            "Sara lives in USA, Washington - Sara, Lucy lives in Turkey, Ankara - Lucy, Tom lives in Italy, Rome - Tom",
            "John lives in Spain, Madrid - John, Michael lives in Canada, Toronto - Michael, Anna lives in Australia, Canberra - Anna",
            "David lives in Japan, Tokyo - David, Sara lives in Brazil, Rio de Janeiro - Sara, Alice lives in France, Paris - Alice",
            "Bob lives in Germany, Berlin - Bob, Peter lives in USA, Washington - Peter, Lucy lives in Turkey, Ankara - Lucy",
            "Anna lives in Italy, Rome - Anna, Tom lives in Spain, Madrid - Tom, David lives in Canada, Toronto - David",
            "Michael lives in Australia, Canberra - Michael, John lives in Japan, Tokyo - John, Sara lives in Brazil, Rio de Janeiro - Sara",
            "Alice lives in France, Paris - Alice, Bob lives in Germany, Berlin - Bob, John lives in USA, Washington - John",
            "Peter lives in Turkey, Ankara - Peter, Alice lives in Italy, Rome - Alice, Bob lives in France, Paris - Bob",
            "Lucy lives in Spain, Madrid - Lucy, Michael lives in Canada, Toronto - Michael, Tom lives in Australia, Canberra - Tom",
            "Anna lives in Japan, Tokyo - Anna, Sara lives in Brazil, Rio de Janeiro - Sara, David lives in France, Paris - David",
            "John lives in Germany, Berlin - John, Peter lives in USA, Washington - Peter, Lucy lives in Turkey, Ankara - Lucy",
            "Tom lives in Italy, Rome - Tom, David lives in Spain, Madrid - David, Michael lives in Canada, Toronto - Michael",
            "Sara lives in Australia, Canberra - Sara, Alice lives in Japan, Tokyo - Alice, Bob lives in Brazil, Rio de Janeiro - Bob",
            "Peter lives in France, Paris - Peter, Lucy lives in Germany, Berlin - Lucy, Tom lives in USA, Washington - Tom",
            "David lives in Turkey, Ankara - David, Michael lives in Italy, Rome - Michael, Anna lives in Spain, Madrid - Anna",
            "John lives in Canada, Toronto - John, Sara lives in Australia, Canberra - Sara, Alice lives in Japan, Tokyo - Alice"
        ]

def process_prompt(prompt):
    # entries = prompt.split(", ")
    data = []
    # # Find the position of the last occurrence of " - "
    last_separator_index = prompt.rfind(" - ")

    # # Slice the string to exclude the last " - Peter"
    entries = prompt[:last_separator_index].split("  ")

    # # Split the sliced prompt into entries
    # entries = sliced_prompt.split(", ")
    for idx, entry in enumerate(entries):
        parts = entry.split(" - ")
        subject_info = parts[0].split(" lives in ")
        subject = subject_info[0]
        attribute = subject_info[1]

        data.append({
            "known_id": idx,
            "subject": subject,
            "attribute": attribute,
            "template": "{} lives in, {} - {}",
            "prediction": subject,
            "prompt": f"{prompt} -",
            "relation_id": f"{last_separator_index}"
        })

    return data

def process_all_prompts(prompts):
    all_data = []
    for prompt in prompts:
        all_data.extend(process_prompt(prompt))
    return all_data

# Process all prompts
all_data = process_all_prompts(prompts)

# Convert all_data to JSON string for pretty printing
all_data_json = json.dumps(all_data, indent=2)
print(all_data_json)

# Specify the directory and file name
output_dir = "."
output_file = "kbicr_data.json"

# Create the directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Path to the output file
output_path = os.path.join(output_dir, output_file)

# Save the processed data to a JSON file
with open(output_path, "w") as file:
    json.dump(all_data, file, indent=2)

print(f"Processed data has been saved to {output_path}")