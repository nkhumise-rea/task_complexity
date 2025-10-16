from pathlib import Path
import os

# directory = Path('/home/rea/pic/examples/data_results/two_link_d')
directory = Path('/home/rea/pic/examples/data_results/1link_100_d_T15')
extension = ".npy"

n_link =  1 #2
t_rew = 'dense' #'sparse'
length = 100 #170

i = 0
for file_path in directory.glob(f"*{extension}"):
    new_name = '{}_link_{}_data_{}_{}.npy'.format(n_link,t_rew,i,length)
    print('old_name: \t', file_path.name)
    print('new_name: \t', new_name)
    os.rename(file_path.name,new_name)
    i += 1