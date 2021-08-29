# -*- coding: utf-8 -*-
"""
@author: ryan.shea
"""


import nltk
from math import log


# read text files

with open("training_data.txt", encoding='utf8') as inp:
    train_file = inp.read()

train = nltk.word_tokenize(train_file)
train_sent = nltk.sent_tokenize(train_file)

with open("test_data.txt", encoding='utf8') as inp:
    test_file = inp.read()

test = nltk.word_tokenize(test_file)
test_sent = nltk.sent_tokenize(test_file)


# get ngram counts

def get_counts(corpus_sent, n):
    count_dict = {}
    for sentence in corpus_sent:
        words = nltk.word_tokenize(sentence)
        for i in range(n-1):
            words.insert(0, "<s>")
        words.append("</s>")
        for i in range(len(words)-n+1):
            gram = []
            for c in range(n):
                gram.append(words[i+c])
            gram_tup = tuple(gram)
            if gram_tup in count_dict.keys():
                count_dict[gram_tup] += 1
            else:
                count_dict[gram_tup] = 1
    print("group complete")
    return count_dict


unigram_counts = get_counts(train_sent, 1)
bigram_counts = get_counts(train_sent, 2)
trigram_counts = get_counts(train_sent, 3)
quadgram_counts = get_counts(train_sent, 4)



def get_count_dict(n):
    if n == 1: return unigram_counts
    if n == 2: return bigram_counts
    if n == 3: return trigram_counts
    if n == 4: return quadgram_counts


# find log probailities for ngrams

def get_ngram_log_probs(n, k=1):
    type_count=len(set(train))
    num_dict=get_count_dict(n)
    num_list=list(num_dict.keys())
    den_dict=get_count_dict(n-1)
    prob_dict={}
    for i in num_list:
        num=num_dict[i]
        context=i[:-1]
        if len([j for j in list(context) if j != '<s>']) == 0:
            den=len(train_sent)
        else:
            den=den_dict[context]
        prob=(num+k)/(den+k*type_count)
        prob_dict[i]=log(prob, 2)
    
    print('done')
    return prob_dict


bi_gram_log_probs=get_ngram_log_probs(2)
tri_gram_log_probs=get_ngram_log_probs(3)
four_gram_log_probs=get_ngram_log_probs(4)


def get_dict(n):
    if n == 2: return bi_gram_log_probs
    if n == 3: return tri_gram_log_probs
    if n == 4: return four_gram_log_probs

test_2_counts = get_counts(test_sent, 2)
test_3_counts = get_counts(test_sent, 3)
test_4_counts = get_counts(test_sent, 4)


# compute perplexity over the test set to find the best ngram model

def get_perplex(test_counts, n, k=1):
    train_dict=get_dict(n)
    
    train_keys=list(train_dict.keys())
    test_keys=list(test_counts.keys())
    matching_keys=set(test_keys).intersection(set(train_keys))
    matching_dict={key: value for key, value in test_counts.items() if key in matching_keys}
    
    
    unknown_prob=k/(len(set(train))*k)
    unknown_count=sum(test_counts.values())-sum(matching_dict.values())
    
    prob_list=[log(unknown_prob,2)*unknown_count*-1]
    
    for i in matching_keys:
        gram=tuple(i)
        prob=train_dict[gram]
        count=test_counts[gram]
        prob_list.append(prob*count*-1)
    
    per_word_cross_entropy=sum(prob_list)/len(test)
    
    return 2**per_word_cross_entropy


perplex_2=get_perplex(test_2_counts, 2)
perplex_3=get_perplex(test_3_counts, 3)
perplex_4=get_perplex(test_4_counts, 4)


with open("unordered.txt", encoding='utf8') as inp:
    scrambled_file = inp.read()

scrambled=nltk.line_tokenize(scrambled_file)


# functions to reorder text using a beam search


def unscramble_line_beam(line, n=2, b=2, k=1):
    words=nltk.word_tokenize(line)
    start_tags=[]
    for i in range(n-1):
        start_tags.append('<s>')
    if n==2:
        current_state=[[['<s>'], 0]]
    else:
        current_state=[[start_tags, 0]]
    res=unscramble_recur(current_state, words, n, b, k)
    return res[0][1:]
    

def unscramble_recur(current_state, words, n, b, k):
    prob_dict=get_dict(n)
    unknown_prob=log(k/(len(set(train))*k),2)
    candidates=[]
    if len(current_state[0][0])>=len(words)+1:
        probs=[i[1] for i in current_state]
        best_unscramble_index=probs.index(max(probs))
        res=current_state[best_unscramble_index]
        return res
    for state in current_state:
        for word in words:
            if (word in state[0]) & (state[0].count(word)>=words.count(word)):
                continue
            if not isinstance(state[0], list):
                gram=[state[0][-(n-1):]]+[word]
            else:
                gram=state[0][-(n-1):]+[word]
            if tuple(gram) in prob_dict.keys():
                prob=prob_dict[tuple(gram)]
            else:
                prob=unknown_prob
            if not isinstance(state[0], list):
                candidates.append([[state[0]]+[word], state[1]+prob])
            else:
                candidates.append([state[0]+[word], state[1]+prob])
    
    unique_candidates=[]
    [unique_candidates.append(i) for i in candidates if i not in unique_candidates] 
    
    probs=[i[1] for i in unique_candidates]
    best_candidate_indices=[]
    for i in range(b):
        good_index=probs.index(max(probs))
        best_candidate_indices.append(good_index)
        probs[good_index]=-100000000
    new_current_state=[unique_candidates[i] for i in best_candidate_indices]
    return unscramble_recur(new_current_state, words, n, b, k)


            
def unscramble_text(lines, n):
    unscrambled=[]
    for line in lines:
        unscramble=unscramble_line_beam(line, n)
        unscrambled.append(unscramble)
    
    return unscrambled


unscrambled=unscramble_text(scrambled, 2)


