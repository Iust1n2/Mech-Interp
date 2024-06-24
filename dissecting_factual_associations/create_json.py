import json
import os 

prompts = [
            "Alice lives in France, Paris - Alice, John lives in Germany, Berlin - John, Peter lives in USA, Washington -",
            "Lucy lives in Turkey, Ankara - Lucy, Sara lives in Italy, Rome - Sara, Bob lives in Spain, Madrid - ",
            "Tom lives in Canada, Toronto - Tom, Anna lives in Australia, Canberra - Anna, Michael lives in Japan, Tokyo - ",
            "David lives in Brazil, Rio de Janeiro - David, Alice lives in France, Paris - Alice, Peter lives in Germany, Berlin - ",
            "Sara lives in USA, Washington - Sara, Lucy lives in Turkey, Ankara - Lucy, Tom lives in Italy, Rome - ",
            "John lives in Spain, Madrid - John, Michael lives in Canada, Toronto - Michael, Anna lives in Australia, Canberra - ",
            "David lives in Japan, Tokyo - David, Sara lives in Brazil, Rio de Janeiro - Sara, Alice lives in France, Paris - ",
            "Bob lives in Germany, Berlin - Bob, Peter lives in USA, Washington - Peter, Lucy lives in Turkey, Ankara - ",
            "Anna lives in Italy, Rome - Anna, Tom lives in Spain, Madrid - Tom, David lives in Canada, Toronto - ",
            "Michael lives in Australia, Canberra - Michael, John lives in Japan, Tokyo - John, Sara lives in Brazil, Rio de Janeiro - ",
            "Alice lives in France, Paris - Alice, Bob lives in Germany, Berlin - Bob, John lives in USA, Washington - ",
            "Peter lives in Turkey, Ankara - Peter, Alice lives in Italy, Rome - Alice, Bob lives in France, Paris - ",
            "Lucy lives in Spain, Madrid - Lucy, Michael lives in Canada, Toronto - Michael, Tom lives in Australia, Canberra - ",
            "Anna lives in Japan, Tokyo - Anna, Sara lives in Brazil, Rio de Janeiro - Sara, David lives in France, Paris - ",
            "John lives in Germany, Berlin - John, Peter lives in USA, Washington - Peter, Lucy lives in Turkey, Ankara - ",
            "Tom lives in Italy, Rome - Tom, David lives in Spain, Madrid - David, Michael lives in Canada, Toronto - ",
            "Sara lives in Australia, Canberra - Sara, Alice lives in Japan, Tokyo - Alice, Bob lives in Brazil, Rio de Janeiro - ",
            "Peter lives in France, Paris - Peter, Lucy lives in Germany, Berlin - Lucy, Tom lives in USA, Washington - ",
            "David lives in Turkey, Ankara - David, Michael lives in Italy, Rome - Michael, Anna lives in Spain, Madrid - ",
            "John lives in Canada, Toronto - John, Sara lives in Australia, Canberra - Sara, Alice lives in Japan, Tokyo - "
        ]

def process_prompt(prompt):
    # Split the entries by " - "
    entries = prompt.split(" - ")
    data = []

    # Loop through the entries except the last one
    for idx in range(len(entries) - 1):
        entry = entries[idx]
        parts = entry.split(" lives in ")
        if len(parts) < 2:
            continue

        subject_info = parts[0].strip()
        attribute_info = parts[1].split(", ")

        if len(attribute_info) < 2:
            continue

        country = attribute_info[0].strip()
        city = attribute_info[1].strip() if len(attribute_info) > 1 else ""

        data.append({
            "known_id": idx,
            "subject": subject_info,
            "attribute": f"{country}, {city}",
            "predicate": "lives in",
            "template": "{} lives in, {} - {}",
            "prediction": "",
            "prompt": prompt,
            "relation_id": f"{idx}"
        })

    # Process the last entry separately
    last_entry = entries[-1].split(", ")
    if len(last_entry) < 2:
        return data

    last_city = last_entry[-1].strip()
    last_country = last_entry[-2].strip()
    last_subject = last_country.split(" ")[-1].strip()
    last_country = " ".join(last_country.split(" ")[:-1]).strip()

    data.append({
        "known_id": idx,
        "subject": subject_info,
        "attribute": f"{country}, {city}",
        "predicate": "lives in",
        "template": "{} lives in, {} - {}",
        "prediction": "",
        "prompt": prompt,
        "relation_id": f"{idx}"
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