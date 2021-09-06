import os
import shutil
import tqdm

origin_folder = '/h/helen/datasets/tag_mask'
#'/h/helen/datasets/frankascan_v2_grnet'
destination_folder = '/h/helen/datasets/frankascan_v2'
file_name = 'tag_mask.png'

# Origin folder names
origin_folder_names = os.listdir(origin_folder)
for subfolder in origin_folder_names:
    # if folder.endswith('.zip'):
    #     continue
    # print(f'Processing {folder}')

    # # Process subfolders
    # subfolders = sorted(os.listdir(os.path.join(origin_folder,folder)),key = lambda x: float(x))
    # for subfolder in subfolders:
    full_subfolder_path = os.path.join(origin_folder,subfolder)
    if file_name in os.listdir(full_subfolder_path):
        # Copy to destination folder path
        if os.path.exists(os.path.join(destination_folder,'test',subfolder)):
            destination_path = os.path.join(destination_folder,'test',subfolder,file_name)
            shutil.copyfile(os.path.join(full_subfolder_path,file_name), destination_path)
            print(f'File copied from {os.path.join(full_subfolder_path,file_name)} to {destination_path}')
        else: 
            destination_path = os.path.join(destination_folder,'val',subfolder,file_name)
            shutil.copyfile(os.path.join(full_subfolder_path,file_name), destination_path)
            print(f'File copied from {os.path.join(full_subfolder_path,file_name)} to {destination_path}')
    else:
        print(os.listdir(full_subfolder_path))
        raise ValueError(f'Filename {file_name} not found in {full_subfolder_path}')
'''
# Origin folder names
origin_folder_names = os.listdir(origin_folder)
for folder in origin_folder_names:
    if folder.endswith('.zip'):
        continue
    print(f'Processing {folder}')

    # Process subfolders
    subfolders = sorted(os.listdir(os.path.join(origin_folder,folder)),key = lambda x: float(x))
    for subfolder in subfolders:
        full_subfolder_path = os.path.join(origin_folder,folder,subfolder)
        if file_name in os.listdir(full_subfolder_path):
            # Copy to destination folder path
            if os.path.exists(os.path.join(destination_folder,folder,subfolder)):
                destination_path = os.path.join(destination_folder,folder,subfolder,file_name)
                shutil.copyfile(os.path.join(full_subfolder_path,file_name), destination_path)
                print(f'File copied from {os.path.join(full_subfolder_path,file_name)} to {destination_path}')
            else: 
                destination_path = os.path.join(destination_folder,'val',subfolder,file_name)
                shutil.copyfile(os.path.join(full_subfolder_path,file_name), destination_path)
                print(f'File copied from {os.path.join(full_subfolder_path,file_name)} to {destination_path}')
        else:
            print(os.listdir(full_subfolder_path))
            raise ValueError(f'Filename {file_name} not found in {full_subfolder_path}')
'''

