import os
import csv

def create_image_csv():
   
    script_dir = os.path.dirname(os.path.abspath(__file__))


    image_dir = os.path.join(script_dir, "..", "data", "images")
    output_csv = os.path.join(script_dir, "..", "data", "csv", "images_labels.csv")

    rows = []

    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
            if 'HH' in filename:
                label = 'HH'
            elif 'Normal' in filename:
                label = 'Normal'
            else:
                continue

            image_path = os.path.join(image_dir, filename)
            rows.append([image_path, label])

    # Ensure csv folder exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'class'])
        writer.writerows(rows)

    print(f"CSV created at: {output_csv}")

create_image_csv()