import os
import math

#The following function is used to read text file and split line based on space
def parsing_file(path):
    word_list=[]
    f= open(path,encoding='utf-8',errors='ignore')
    for l in f:
        for aux in l.split():
            word_list.append(aux)
    return word_list

#Storing ham and spam words in the dictionary based on key
def ham_spam_data_store():
    spam_ham_dict = {'ham': [], 'spam': []}
    count={"ham":0,"spam":0}
    for fN1 in os.listdir(os.getcwd() + "/test/ham" ):
        wds = parsing_file(os.getcwd() +  "/test/ham/" + fN1)
        if len(wds) > 0:
            spam_ham_dict['ham'].append(wds)
        count['ham'] += 1
    for fN2 in os.listdir(os.getcwd() + "/test/spam"):
        words = parsing_file(os.getcwd() + "/test/spam/" + fN2)
        if len(words) > 0:
            spam_ham_dict['spam'].append(words)
        count['spam'] += 1
    return spam_ham_dict
def stop_word_creation(spam_ham_dict):
    word_list=[]
    for d1 in spam_ham_dict:
        for info in spam_ham_dict[d1]:
            for w in info:
                if w not in word_list:
                    word_list.append(w)
    return word_list
                
def without_stop_word_creation(spam_ham_dict, stop_word_list):
    word_list=[]
    for d2 in spam_ham_dict:
        for info in spam_ham_dict[d2]:
            for w in info:
                if w not in word_list and w.lower() not in stop_word_list :
                    word_list.append(w)
    return word_list

def sum_Total(inputs,weight):
    weight_sum=0.0
    for f,val in inputs.items():
        if f in weight:
            weight_sum=weight_sum+(val*weight[f])
    return weight_sum
 
def prob_cls(inputs,weight):
    tot_sum=sum_Total(inputs,weight)
    try:
        val=math.exp(tot_sum)*1.0
    except OverflowError:
        return 1
    return round(val/(1.0+val),5)

def Logit_Reg_training(spam_ham_dict,word_list,I,l_rate,l):
    weight={'bias':0.0}
    for w in word_list:
        weight[w]=0.0
    for i in range (0,I):
        err_summer={}
        for di in spam_ham_dict:
            for data in spam_ham_dict[di]:
                inputs = {'bias': 1.0}
                for d in data:
                    inputs[d]=data.count(d)
                err = c_val[di] - prob_cls(inputs, weight)
                if err!=0:
                    for f in inputs:
                        if f in err_summer:
                            err_summer[f]+=(inputs[f]*err)
                        else:
                            err_summer[f]=(inputs[f]*err)
        for w in weight:
            if w in err_summer:
                weight[w]=weight[w]+(l_rate*err_summer[w])-(l_rate*weight[w])
    return weight

def features(fpath,fname):
    inputs={'bias':1.0}
    words=parsing_file(fpath+"/"+fname)
    for w in words:
        inputs[w]=words.count(w)
    return inputs
    
def logit_reg_test(weight):
    acc={1:0.0,0:0.0}
    for files in os.listdir(os.getcwd() + "/test/" + 'ham'):
        feature = features(os.getcwd() + "/test/" + 'ham',files)
        cls_sum=sum_Total(feature,weight)
        if(cls_sum>=0):
            acc[1]+=1.0
        else:
            acc[0]+=1.0
    for files in os.listdir(os.getcwd() + "/test/" + 'spam'):
        feature = features(os.getcwd() + "/test/" + 'spam',files)
        cls_sum=sum_Total(feature,weight)
        if(cls_sum<0):
            acc[1]+=1.0
        else:
            acc[0]+=1.0
    return ((acc[1]*100)/sum(acc.values()))

#testing
count= {"ham": 0.0, "spam": 0.0}
c_val = {'ham': 1.0, 'spam': 0.0}
addr_files=os.getcwd() + "/test/stop_words.txt"
stop_word_list=parsing_file(addr_files)
spam_ham_dict=ham_spam_data_store()

info_word_with_stop_words=stop_word_creation(spam_ham_dict)
info_word_without_stop_words=without_stop_word_creation(spam_ham_dict,stop_word_list)
    
learning_rate = 0.001
lam = {0.01,0.2,0.5,1,1.5}
i = 50
for l in lam:
    weight = Logit_Reg_training(spam_ham_dict, info_word_with_stop_words, i, learning_rate, l)
    print("When learning value is:", l)
    print("LR Accuracy including Stop Words : " + str(logit_reg_test(weight)))
    weight = Logit_Reg_training(spam_ham_dict, info_word_without_stop_words, i, learning_rate, l)
    print("LR Accuracy removing Stop Words : " + str(logit_reg_test(weight)))

