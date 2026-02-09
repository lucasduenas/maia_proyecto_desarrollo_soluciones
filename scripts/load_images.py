import os
import csv

def create_image_csv():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))

    image_dir = os.path.join(project_root, "data", "images")
    output_csv = os.path.join(project_root, "data", "csv", "images_labels.csv")

    rows = []

    # Recorrer carpetas de labels
    for label in os.listdir(image_dir):
        label_path = os.path.join(image_dir, label)

        if not os.path.isdir(label_path):
            continue

        for filename in os.listdir(label_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                image_path = os.path.join(label_path, filename)

                # Ruta relativa respecto al root del proyecto
                relative_path = os.path.relpath(image_path, project_root)

                rows.append([relative_path, label])

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'class'])
        writer.writerows(rows)

    print(f"CSV created at: {output_csv}")

create_image_csv()