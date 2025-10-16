from pathlib import Path
import os

# directory = Path('/home/rea/pic/examples/data_results/two_link_d')
directory = Path('/home/rea/pic/examples/data_results/one_link_s')
extension = ".npy"

n_link =  1 #2
t_rew = 'sparse' #'dense'

i = 0
for file_path in directory.glob(f"*{extension}"):
    new_name = '{}_link_{}_data_{}.npy'.format(n_link,t_rew,i)
    print('old_name: \t', file_path.name)
    print('new_name: \t', new_name)
    os.rename(file_path.name,new_name)
    i += 1