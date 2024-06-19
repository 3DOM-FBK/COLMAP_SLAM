import os

folder_path = r'C:\Users\fbk3d\Desktop\buttare_fixposition\images\cam1\data'

# Get all the files in the folder
file_list = os.listdir(folder_path)

# Iterate over each file
for file_name in file_list:
    # Construct the new file name by eliminating the first 5 characters
    new_file_name = file_name[6:]
    
    # Rename the file
    os.rename(os.path.join(folder_path, file_name), os.path.join(folder_path, new_file_name))