import os
import shutil
log_dir = 'log/part_seg'
logs = os.listdir(log_dir)
for log in logs:
    log_folder_path = os.path.join(log_dir,log)
    best_model_path = os.path.join(log_folder_path,'checkpoints','best_model.pth')
    if os.path.exists(best_model_path):
        pass
    else:
        shutil.rmtree(log_folder_path)
print('clean over')


