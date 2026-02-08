import os
import csv

def create_image_csv():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Raíz del proyecto (un nivel arriba del script)
    project_root = os.path.abspath(os.path.join(script_dir, ".."))

    image_dir = os.path.join(project_root, "data", "images")
    output_csv = os.path.join(project_root, "data", "csv", "images_labels.csv")

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

            # Convertir a ruta relativa respecto a project_root
            relative_path = os.path.relpath(image_path, project_root)

            rows.append([relative_path, label])

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'class'])
        writer.writerows(rows)

    print(f"CSV created at: {output_csv}")

create_image_csv()