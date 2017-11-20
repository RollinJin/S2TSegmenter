'''
Created on Nov 7, 2017

@author: rollinjin
'''
import json
from keras.models import load_model
import numpy as np
import nltk

model_path = "../model/model.s2t"
flags = ['', ',', '.']

def process_events(fpath):
    all_words = []
    all_pauses = []
   
    model=load_model(model_path)
    
    with open(fpath) as f:
        outputs = json.load(f)
        for segment in outputs:
            words = segment['words']
            #all_words = np.concatenate((all_words,words),axis=0)
            all_words = all_words + words
            
    #print(all_words) 
    
    pre_avg_time = 0.0
    pre_word_length = 0;
    pre_word_totaltime = 0.0
    word_num = len(all_words)
    total_wordtime = 0.0
    
    min_pause = 0.05
    max_pause = 2.0
    
    word_pause_idx = []
        
    #[Overall average pause, overall average Word time, current pause, average word time of previous words, word length between previous pause]
    for i in range(0, word_num):
        word_pause_idx.append(-1)
        word = all_words[i]
        word_time = word[2] - word[1]
        total_wordtime += word_time
        
        if i<word_num-1:        
            next_word = all_words[i+1]
            cur_pause = round(next_word[1] - word[2], 2)
        else:
            cur_pause = max_pause
        
        if cur_pause>min_pause:
            word_pause = []
            if pre_word_length>0:
                pre_avg_time = round(pre_word_totaltime/pre_word_length, 2)
                
            word_pause.append(cur_pause)
            word_pause.append(pre_avg_time)
            word_pause.append(pre_word_length/10.0)
            
            word_pause_idx[i] = len(all_pauses) #Record the pause index for this word
            all_pauses.append(word_pause)
            pre_avg_time = 0.0
            pre_word_length = 0;
            pre_word_totaltime = 0.0
        else:
            pre_word_length += 1 
            pre_word_totaltime += word_time 
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
    
    #print(new_all_pauses)
    output_str = ""
    
    predicts_output = model.predict(new_all_pauses, batch_size=128, verbose=1) 
    #print(predicts_output)
    predicts = np.argmax(predicts_output)
    #print(predicts)
    
    
    if len(all_words[0])==4:
        labelFlag = True
    else:
        labelFlag = False
        
        
    for j in range(0, word_num):
        if j<word_num-1 and labelFlag and all_words[j][3]!=all_words[j+1][3]:
            periodFlag = True
        else:
            periodFlag = False
                
        if word_pause_idx[j]==-1:
            output_str = output_str + all_words[j][0]  
            if periodFlag:
                output_str = output_str + "."            
            output_str = output_str + " "
        else:
            pause_idx = word_pause_idx[j]     
            predict = np.argmax(predicts_output[pause_idx])
            #print(predict)
            if periodFlag:
                output_str = output_str + all_words[j][0] + ". "
            else:               
                output_str = output_str + all_words[j][0] + flags[predict] + " "  
   
    
    segments = nltk.sent_tokenize(output_str)
    for segment in segments:
        newStr = segment
        if segment[0].isalpha():
            newStr = segment[0].upper() + segment[1:]
        print(newStr)
#fpath = "../corpus/json/Blockchain_events.json"
#fpath = "../corpus/json/Watson_VR_events.json"
#fpath = "../corpus/json/Watson_Discovery_events.json"
fpath = "../corpus/json/Car_events.json"
#fpath = "../corpus/json/SETI_Institute_events.json"
process_events(fpath)