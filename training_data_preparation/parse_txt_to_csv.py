#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 11:08:48 2022

@author: xingangli
"""

import csv

perioded_txt_file = 'period.txt'
non_perioed_txt_file = 'no-period.txt'
data = 'data.csv'
data2 = 'data2.csv'

#function for formatting 
#perioded txt file to csv file
def perioded_txt_to_csv(txt_file, csv_file):

    with open(txt_file) as file:
        txt = file.readlines()
    
    #strip all "\n"
    txt_list = []
    for line in txt:
        line = line.strip()
        txt_list.append(line)
    
    #concatenate all strings to a whole paragraph
    all_txt_string = ' '.join(txt_list)
    
    #split the paragraph by "."
    txt_list_new = all_txt_string.split(".")
    
    #form the format of [index, string]
    txt = []
    i = 1
    for sentence in txt_list_new:
        if len(sentence) > 30: #fileter out something like ".py", ".com"
            txt.append([i, sentence.strip()])
            i += 1
    
    print(len(txt))
        
    #form the csv file for training
    header = ['ID', 'data']
    with open(csv_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(txt)

        
#function for formatting 
#non-perioded txt file to csv file
#concatenate n lines as one sentence
def txt_to_csv(txt_file, csv_file):

    with open(txt_file) as file:
        txt = file.readlines()
    
    #strip all "\n"
    txt_list = [line.strip() for line in txt if ((line != '\n') and (line.strip() != '[Music]'))]
    # txt_list = [line.strip() for line in txt if line.strip() == '[Music]']
    # print(len(txt_list))
    
    #concate every n lines
    txt = []
    n = 5
    for i in range(len(txt_list) // n):
        cell_data = txt_list[5*i + 0] + ' ' + txt_list[5*i + 1] + ' ' \
                  + txt_list[5*i + 2] + ' ' + txt_list[5*i + 3] + ' ' \
                  + txt_list[5*i + 4]
        #start index from 9976
        txt.append([9976 + i, cell_data]) #perioed data index end at 9975

    print(len(txt))          
        
    #form the csv file for training
    header = ['ID', 'data']
    with open(csv_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(txt)    
              
perioded_txt_to_csv(perioded_txt_file, data)
txt_to_csv(non_perioed_txt_file, data2)