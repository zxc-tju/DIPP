import os
import zipfile

def zip_folder(folder_path, output_path):
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, arcname)

folder_path = '/mnt/workspace/codes/DIPP/training_log'  # Replace with the path to your folder
output_path = '/mnt/workspace/DIPP_training_log_from_clould_DDM.zip'  # Replace with the desired output path

zip_folder(folder_path, output_path)
