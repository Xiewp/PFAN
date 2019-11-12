#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 17:05:04 2019

@author: smartdsp
"""

with open('webcam_list.txt','r') as f1:
    f_c = f1.readlines()
filename = []
for j in f_c:
    filename.append(j[:-3])

dic_target = {}
with open('pre_and_sim.txt','r') as f:
    f_context = f.readlines()
count = 0
for i in f_context:
    dic_target.setdefault(eval(i.split(' ')[0]),[]).append([eval(i.split(' ')[1]),filename[count]])
    count += 1
class_c = 0
result = []
for k in range(31):
    if k not in dic_target:
        continue
    ans = sorted(dic_target[k],reverse = True)
    if 0 < len(ans) <= 3:
        for t in ans:
            result.append(t[1] + ' ' + str(k))
    else:
        for t in range(3):
            result.append(ans[t][1] + ' ' + str(k))
with open('amazon_list.txt','r') as fs:
    source = fs.readlines()
with open('a_with_pseudo.txt','w') as fn:
    for i in source:
        fn.write(i)
    for j in result:
        fn.write(j+'\n')
print('ok')