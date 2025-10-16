import scipy.stats as stats
import numpy as np

pic_1link100_d = np.array([
                3.974,
                3.953,
                3.951,
                3.960,
                3.962
                ])

pic_1link165_d = np.array([
                4.111,
                4.105,
                4.122,
                4.121,
                4.088
                ])

pic_2link_d = np.array([
                4.207,
                4.191,
                4.195,
                4.171,
                4.175,
                ])

pic_1link100_s = np.array([
                0.08577,
                0.08344,
                0.08556,
                0.08575,
                0.08458
                ])
pic_1link165_s = np.array([
                0.07066,
                0.07225,
                0.07078,
                0.07126,
                0.07168])

pic_2link_s = np.array([
            0.04573,
            0.04599,
            0.04606,
            0.04604,
            0.04583
            ])


groups = [pic_1link100_d,
          pic_1link165_d,
          pic_2link_d,
          pic_1link100_s,
          pic_1link165_s,
          pic_2link_s]

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
        print('PIC: {} vs {} = '.format(group_name[i],group_name[k]), jj)
        print('_____________________')
        # xxx