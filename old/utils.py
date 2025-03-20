# -*- coding: utf-8 -*-
"""
@author: DimiP
"""

from numpy import genfromtxt


def csv2numpy(csv_url):
    my_data = genfromtxt(csv_url, delimiter=',')
    data4col=np.zeros((71,4))
    for i in range(1,71):
        data4col[i][0]=my_data[i][1]
        data4col[i][1]=my_data[i][3]
        data4col[i][2]=my_data[i][5]
        data4col[i][3]=my_data[i][6]
    return data4col

