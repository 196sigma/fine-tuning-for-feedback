import json
import jsonlines

def deduplication_jsonlines(infile:str, outfile:str=None):
    """Remove duplicate jsonlines from `infile` and write to `outfile`."""
    if not outfile:
        outfile = infile
    
    with jsonlines.open(infile) as reader:
        samples = []
        for obj in reader:
            samples.append(obj)
    print(f"Loaded {len(samples)} samples from {infile}")
    
    dedup_samples = []
    for sample in samples:
        if sample not in dedup_samples:
            dedup_samples.append(sample)

    print(f"Deduplicated {len(dedup_samples)} unique samples")
    
    with jsonlines.open(outfile, 'w') as writer:
        for sample in dedup_samples:
            writer.write(sample)

def clean_jsonlines(infile:str, outfile:str=None):
    """Make sure each line of jsonlines `infile` is valid json."""
    if not outfile:
        outfile = infile
    
    with open(infile, 'r') as reader:
        lines = reader.readlines()
    
    with open(outfile, 'w') as writer:
        for line in lines:
            try:
                obj = json.loads(line)
                try:
                    keys = obj.keys()
                    if list(keys) == ['summary', 'feedback']:
                        writer.write(line)
                except AttributeError:
                    print(f"Attr Error: {line}")
            except json.JSONDecodeError:
                print(f"JSON Error: {line}")

def jsonl_to_json(jsonl_filepath:str, json_filepath:str):
    """Function to convert jsonl to json."""
    data_list = []

    # Read the JSON lines file
    with open(jsonl_filepath, 'r') as file:
        for line in file:
            # Convert each line to a JSON object and append to the list
            json_obj = json.loads(line)
            data_list.append(json_obj)

    # Write the list to a JSON file
    with open(json_filepath, 'w') as json_file:
        json.dump(data_list, json_file, indent=4)

    print(f"Converted {jsonl_filepath} to {json_filepath}")
