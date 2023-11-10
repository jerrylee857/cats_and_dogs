import os
import random
import shutil

def move_files_to_validation(src_folder, dest_folder, ratio=0.1):
    categories = ['cats', 'dogs']

    for category in categories:
        src_path = os.path.join(src_folder, category)
        dest_path = os.path.join(dest_folder, category)

        if not os.path.exists(dest_path):
            os.makedirs(dest_path)

        all_files = os.listdir(src_path)
        random.shuffle(all_files)
        
        num_to_move = int(len(all_files) * ratio)

        files_to_move = all_files[:num_to_move]

        for filename in files_to_move:
            src_filepath = os.path.join(src_path, filename)
            dest_filepath = os.path.join(dest_path, filename)

            shutil.move(src_filepath, dest_filepath)

# 使用示例
move_files_to_validation('cats_and_dogs_train', 'cats_and_dogs_valid')
