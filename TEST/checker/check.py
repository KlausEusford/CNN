# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 21:39:38 2017

@author: zhouyi
"""

gold = open("gold.txt")
pred = open("pred.txt")

corr = 0

while True:
    gold_line = gold.readline()
    pred_line = pred.readline()
    if gold_line == '':
        break
    if int(gold_line) == int(pred_line):
        corr += 1
    
print("accuracy {}".format((float) (corr*1.0/763)))
