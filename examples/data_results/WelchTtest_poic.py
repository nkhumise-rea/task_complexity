import scipy.stats as stats
import numpy as np

poic_1link100_d = np.array([
2.65E-03,
2.79E-03,
2.62E-03,
2.65E-03,
2.41E-03,
                ])

poic_1link165_d = np.array([
4.06E-03,
4.22E-03,
4.01E-03,
4.20E-03,
3.93E-03,
                ])

poic_2link_d = np.array([
7.34E-04,
7.45E-04,
7.10E-04,
6.96E-04,
7.31E-04,
                ])

poic_1link100_s = np.array([
2.03E-03,
1.82E-03,
2.03E-03,
2.11E-03,
1.91E-03,
                ])
poic_1link165_s = np.array([
1.18E-03,
1.26E-03,
1.15E-03,
1.16E-03,
1.21E-03,
                ])

poic_2link_s = np.array([
9.21E-04,
9.62E-04,
9.65E-04,
9.31E-04,
9.57E-04,
            ])


groups = [poic_1link100_d,
          poic_1link165_d,
          poic_2link_d,
          poic_1link100_s,
          poic_1link165_s,
          poic_2link_s]

group_name = ['1link100d',
              '1link165d',
              '2linkd',
              '1link100s',
              '1link165s',
              '2links']

for i,group1 in enumerate(groups):
    # print('i: ', i)
    # print('group1: ', group1)
    # xxxx
    for k,group2 in enumerate(groups):
        jj = stats.ttest_ind(group1,group2,equal_var=False)
        print('poic: {} vs {} = '.format(group_name[i],group_name[k]), jj)
        print('_____________________')
        # xxx