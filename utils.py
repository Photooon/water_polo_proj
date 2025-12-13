import os

from typing import List

def get_imgs_from_dir_or_file(path: str) -> List[str]:
    image_files = []

    if os.path.isdir(path):
        valid_extensions = {'.jpg', '.jpeg', '.png',}
        for filename in sorted(os.listdir(path)):
            ext = os.path.splitext(filename)[1].lower()
            if ext in valid_extensions:
                image_files.append(os.path.join(path, filename))
        # print(f"Found {len(image_files)} images in directory")
    elif os.path.isfile(path):
        image_files.append(path)
    else:
        print(f"Error: {path} is not a valid file or directory")
        exit(1)

    return image_files