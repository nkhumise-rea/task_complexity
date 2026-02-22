from pathlib import Path
import os

directory = Path.cwd() 
extension = ".npy"

n_link =  2
t_rew = 'dense' #'sparse'

i = 0
for file_path in directory.glob(f"*{extension}"):
    print(file_path)
    new_name = '{}_link_{}_data_{}.npy'.format(n_link,t_rew,i)
    print('old_name: \t', file_path.name)
    print('new_name: \t', new_name)
    os.rename(file_path.name,new_name)
    i += 1