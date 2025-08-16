import json
from tqdm import tqdm
import os
import random
def convert_to_vlm_format(data, base_image_path='../JPEGImages'):
    result = []
    for item in tqdm(data):
        
        prompt = (
            """
            You are a remote sensing image understanding and reasoning assistant. Your task is to locate the target object in the image based on the provided text description and output its bounding box coordinates. Please follow these requirements:
            1. Text comprehension: Carefully analyze the text description to identify the target object.\n
            2. Output format:\n
                - If one object exists: Output the bounding box coordinates in the format: [x_min, y_min, x_max, y_max].\n
                - If no matching object is found: Output an empty list [].\n
                - If multi-objects exist: Output the bounding box coordinates in the format: [[x_min1, y_min1, x_max1, y_max1], [x_min2, y_min2, x_max2, y_max2], ...].\n
            3. Coordinate requirements:\n
                - Coordinate values must be integers representing pixel positions.\n
                - Critical: You MUST output the tight bounding box around the object.\n
            4. Output specifications:\n
                - Output the thinking process in <think> </think> and final answer (a list of bbox) in <answer> </answer> tags.\n
                Perform the detection task for the following input:\n
            <image>\n
            Text description: 
            """
        )
            

        # messages
        messages = [
            {
                "role": "user",
                "content": f"{prompt}\n<image>\nText description: {item['text']}"
            }
        ]

        solution = f"<answer> {str(item['mask'])} </answer>"

        
        new_item = {
            "images": os.path.join(base_image_path, item['image']),  # Note: Should be actual image path/URL
            "messages": messages,
            "solution": solution
        }
        result.append(new_item)
    return result


def split_dataset(data, train_ratio=0.9, random_seed=42):
    
    random.seed(random_seed)
 
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)

    split_index = int(len(shuffled_data) * train_ratio)

    train_data = shuffled_data[:split_index]
    val_data = shuffled_data[split_index:]

    return train_data, val_data

with open('./triplets_train.json', 'r', encoding='utf-8') as f:
    remoteSAM_data = json.load(f)

converted_data = convert_to_vlm_format(remoteSAM_data)



train_file = "Train_RL.json"
val_file = "Val_RL.json"

with open(train_file, 'w') as f:
    json.dump(converted_data, f, indent=2)


print(json.dumps(converted_data[0], indent=2))

