'''
Created on Nov 7, 2017

@author: rollinjin
'''

import json
import sys
import numpy as np
import xlrd
import nltk

flags = [',', ':', '.', '?']

root_dir = "../corpus/xls/"

fpath_array = ['../corpus/json/Watson_VR_events.json',
               '../corpus/json/Watson_Discovery_events.json',
               '../corpus/json/Blockchain_events.json']

xls_array = ['Watson_VR.xlsx',
             'Watson_Discovery.xlsx',
             'Blockchain.xls']

array_names = ['../corpus/array/Watson_VR_Data.npy',
               '../corpus/array/Watson_Discovery_Data.npy',
               '../corpus/array/Blockchain_Data.npy']

corpusNum = 3


def load_xls(fname, skip_fisrt_line=False, remove_tstamp=True):
    cell_content = ""
    exlpath = root_dir + fname
    data = xlrd.open_workbook(exlpath)
    table = data.sheets()[0]

    start_idx = 1 if skip_fisrt_line else 0
  
    xls_data = []
    ncols = table.ncols 
    for r in range( start_idx,table.nrows ):   
        try:
            if(len(table.row_values(r)[0])==0 and ncols>1 and len(table.row_values(r)[1])>0):
                cell_content = table.row_values(r)[1]
                #print table.row_values(r)[1]
            else:
                cell_content = table.row_values(r)[0]
            
            if(remove_tstamp):
                if(len(cell_content)>5):                
                    xls_data.append(cell_content[6:])    
            else:
                xls_data.append(cell_content)   
                
            #print(cell_content)   
        except ValueError as e:
            print(e)
            
    xls_data = [line.lower() for line in xls_data if len(line.strip()) ]
    return xls_data


def load_events(fpath):
    all_words = []
    all_pauses = []
    
    with open(fpath) as f:
        outputs = json.load(f)
        for segment in outputs:
            words = segment['words']
            #all_words = np.concatenate((all_words,words),axis=0)
            all_words = all_words + words
            
    print(all_words) 
    
    pre_avg_time = 0.0
    pre_word_length = 0;
    pre_word_totaltime = 0.0
    word_num = len(all_words)
    total_wordtime = 0.0
    
    min_pause = 0.05
    max_pause = 2.0
        
    #[Overall average pause, overall average Word time, current pause, average word time of previous words, word length between previous pause, 
    # average word time of backward words, word length between backward pause]
    for i in range(0, word_num):
        word = all_words[i]
        word_time = word[2] - word[1]
        total_wordtime += word_time
        
        if i<word_num-1:        
            next_word = all_words[i+1]
            cur_pause = round(next_word[1] - word[2], 2)
        else:
            cur_pause = max_pause
        
        pre_word_length += 1 
        pre_word_totaltime += word_time 
        
        if cur_pause>min_pause:
            word_pause = []
            if pre_word_length>0:
                pre_avg_time = round(pre_word_totaltime/pre_word_length, 2)
                
            word_pause.append(cur_pause)
            word_pause.append(pre_avg_time)
            word_pause.append(pre_word_length/10.0)
            word_pause.append(0.0)
            word_pause.append(0.0)
            word_pause.append(word[0])
            
            if len(all_pauses)>0:
                preWordPause = all_pauses[-1]
                preWordPause[3] = pre_avg_time
                preWordPause[4] = pre_word_length/10.0
            
            all_pauses.append(word_pause)
            pre_avg_time = 0.0
            pre_word_length = 0;
            pre_word_totaltime = 0.0
            
        #print(word)
    
    pause_num = len(all_pauses)
    total_pause = 0.0
    for pause in all_pauses:
        total_pause += pause[0]
    
    overall_avg_pause = round(total_pause/pause_num, 2)
    overall_avg_wordtime  = round(total_wordtime/word_num, 2)
    
    pre_list = []
    pre_list.append(overall_avg_pause)
    pre_list.append(overall_avg_wordtime)
    print(pre_list)
    
    new_all_pauses = []
    
    for pause in all_pauses:
        new_all_pauses.append(pre_list+pause)
    
    return new_all_pauses
    
def parse_pause(all_pauses, all_tokens):
    valid_pause = []
    cur_pos = 0
    for vPause in all_pauses:
        word = vPause[-1]
        try:
            idx = all_tokens.index(word, cur_pos)
            if idx>-1 :
                if all_tokens[idx+1] in flags:
                    out_idx = flags.index(all_tokens[idx+1])
                    if out_idx<2 : # comma
                        out_idx=1
                    else: # Period
                        out_idx=2   
                else: # Blank space
                    out_idx = 0                           
                vPause.append(out_idx);    
                valid_pause.append(vPause)
                #if out_idx == 1:
                   # valid_pause.append(vPause) #Double the samples of comma
                #print(word, out_idx) 
                if (idx - cur_pos)>30: 
                    print(str(idx - cur_pos) + ":" + word)          
                else:
                    cur_pos = idx + 1         
        except ValueError as e:
            cur_pos += 0 
            #print(e) 
    return valid_pause



for i in range(0, corpusNum):
    fpath = fpath_array[i]
    xls_fname = xls_array[i]
    array_name = array_names[i]

    all_pauses = load_events(fpath)
    print(all_pauses)
    
    #xls_data = load_xls(xls_fname, False, True)
    xls_data = load_xls(xls_fname, True, False) #Blockchain
    
    all_tokens = []
    for segment in xls_data:
        #print(segment)
        tokens = nltk.word_tokenize(segment)
        all_tokens = all_tokens + tokens
    
    #print(all_tokens)
    
    
    valid_pause = parse_pause(all_pauses, all_tokens)
    #print(valid_pause) 
    valid_pause = np.delete(valid_pause, -2, axis=1)
    print(valid_pause) 
    
    np.save(array_name, valid_pause)   

