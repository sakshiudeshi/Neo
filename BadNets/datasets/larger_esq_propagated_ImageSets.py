import os, sys, time
from os import listdir


larger_esq_propagated_imgs = listdir('./usts/disabled_attack/stop-speedlimit-esq-propagated')
namelist=[]

for i in larger_esq_propagated_imgs:
    name = i.split('.')[0]
    namelist.append(name)
with open('./usts/ImageSets/test_targ_larger-esq-propagated_backdoor.txt', 'w') as f:
    f.write('\n'.join(['%07d'%int(x) for x in namelist]))
