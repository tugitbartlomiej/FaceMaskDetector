import os

def check_and_delete_files(base_dir):
    for subdir, dirs, files in os.walk(base_dir):
        for filename in files:
            filepath = os.path.join(subdir, filename)
            # Process only text files
            if filepath.endswith(".txt"):
                with open(filepath, 'r') as file:
                    first_line = file.readline()
                    if not first_line.startswith('1.0'):
                        os.remove(filepath)
                        print(f"Deleted: {filepath}")

def check_and_modify_files(base_dir):
    for subdir, dirs, files in os.walk(base_dir):
        for filename in files:
            filepath = os.path.join(subdir, filename)
            # Process only text files
            if filepath.endswith(".txt"):
                with open(filepath, 'r') as file:
                    lines = file.readlines()
                # Check if the first line starts with '1.0'
                if lines[0].startswith('1.0'):
                    # Change '1.0' to '0'
                    lines[0] = lines[0].replace('1.0', '0')
                    with open(filepath, 'w') as file:
                        file.writelines(lines)
                    print(f"Modified: {filepath}")

base_directory = '/home/bartlomiej/Studia/Sem4/Przetwarzanie_Obrazów/face-masks/scripts/yolov8/dataset/'
# check_and_delete_files(base_directory)
check_and_modify_files(base_directory)
