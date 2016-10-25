from collections import Counter, defaultdict
from itertools import chain, combinations
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from scipy.sparse import csr_matrix
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
import string
import tarfile
import urllib.request


def download_data():
    url = 'https://www.dropbox.com/s/xk4glpk61q3qrg2/imdb.tgz?dl=1'
    urllib.request.urlretrieve(url, 'imdb.tgz')
    tar = tarfile.open("imdb.tgz")
    tar.extractall()
    tar.close()


def read_data(path):
  fnames = sorted([f for f in glob.glob(os.path.join(path, 'pos', '*.txt'))])
    data = [(1, open(f).readlines()[0]) for f in sorted(fnames)]
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'neg', '*.txt'))])
    data += [(0, open(f).readlines()[0]) for f in sorted(fnames)]
    data = sorted(data, key=lambda x: x[1])
    return np.array([d[1] for d in data]), np.array([d[0] for d in data])


def tokenize(doc, keep_internal_punct=False):
    """
   >>> tokenize(" Hi there! Isn't this fun?", keep_internal_punct=False)
    array(['hi', 'there', 'isn', 't', 'this', 'fun'], 
          dtype='<U5')
    >>> tokenize("Hi there! Isn't this fun? ", keep_internal_punct=True)
    array(['hi', 'there', "isn't", 'this', 'fun'], 
          dtype='<U5')
    """
    t = []
    if keep_internal_punct==False:
        t=re.sub('\W+', ' ', doc.strip().lower()).split()
    elif keep_internal_punct==True:
        t = doc.split()
        t = [i.lower().lstrip('{}'.format(string.punctuation)).rstrip('{}'.format(string.punctuation)) for i in t]
        
    tokens=np.array(t)
    return tokens


def token_features(tokens, feats):
    """
   >>> feats = defaultdict(lambda: 0)
    >>> token_features(['hi', 'there', 'hi'], feats)
    >>> sorted(feats.items())
    [('token=hi', 2), ('token=there', 1)]
    """
    c=Counter(tokens)
    for t in tokens:
        feats["token="+t]=c[t]


def token_pair_features(tokens, feats, k=3):
    """
    >>> feats = defaultdict(lambda: 0)
    >>> token_pair_features(np.array(['a', 'b', 'c', 'd']), feats)
    >>> sorted(feats.items())
    [('token_pair=a__b', 1), ('token_pair=a__c', 1), ('token_pair=b__c', 2), ('token_pair=b__d', 1), ('token_pair=c__d', 1)]
    """
    for i in range(len(tokens)):
        if i+k>len(tokens):
            temp=len(tokens)-1
        else:
            temp=i+k
        for j in range(i,temp-1):
            for l in range(j+1,temp):
                feats["token_pair="+tokens[j]+"__"+tokens[l]]+=1


neg_words = set(['bad', 'hate', 'horrible', 'worst', 'boring'])
pos_words = set(['awesome', 'amazing', 'best', 'good', 'great', 'love', 'wonderful'])

def lexicon_features(tokens, feats):
    """
    >>> feats = defaultdict(lambda: 0)
    >>> lexicon_features(np.array(['i', 'LOVE', 'this', 'great', 'boring', 'movie']), feats)
    >>> sorted(feats.items())
    [('neg_words', 1), ('pos_words', 2)]
    """
    feats["neg_words"]=0
    feats["pos_words"]=0
    for i in tokens:
        if i.lower() in neg_words:
            feats["neg_words"]+=1
        if i.lower() in pos_words:
            feats["pos_words"]+=1


def featurize(tokens, feature_fns):
    """
    >>> feats = featurize(np.array(['i', 'LOVE', 'this', 'great', 'movie']), [token_features, lexicon_features])
    >>> feats
    [('neg_words', 0), ('pos_words', 2), ('token=LOVE', 1), ('token=great', 1), ('token=i', 1), ('token=movie', 1), ('token=this', 1)]
    """
    feats = defaultdict(lambda: 0)
    for i in feature_fns:
        i(tokens,feats)
    return sorted(feats.items())


def vectorize(tokens_list, feature_fns, min_freq, vocab=None):
    """
    >>> docs = ["Isn't this movie great?", "Horrible, horrible movie"]
    >>> tokens_list = [tokenize(d) for d in docs]
    >>> feature_fns = [token_features]
    >>> X, vocab = vectorize(tokens_list, feature_fns, min_freq=1)
    >>> type(X)
    <class 'scipy.sparse.csr.csr_matrix'>
    >>> X.toarray()
    array([[1, 0, 1, 1, 1, 1],
           [0, 2, 0, 1, 0, 0]], dtype=int64)
    >>> sorted(vocab.items(), key=lambda x: x[1])
    [('token=great', 0), ('token=horrible', 1), ('token=isn', 2), ('token=movie', 3), ('token=t', 4), ('token=this', 5)]
    """
    def generate_vocab():
        i=0
        for f in sorted(freq):
            if freq[f]>=min_freq:
                vocab[f]=i
                i+=1
    dicts=[]
    freq={}
    
    data_row=[]
    data=[]
    data_col=[]
    count=0
    for doc in tokens_list:
        temp_feats={}
        lis=featurize(doc, feature_fns)
        for a in lis:
            if a[1]>0:
                temp_feats[a[0]]=a[1]
        dicts.append(temp_feats)
        for feature in lis:
            if feature[1]>0:
                if feature[0] in freq:
                    freq[feature[0]]+=1
                else:
                    freq[feature[0]]=1
    if vocab != None:
        for doc in dicts:
            for f in doc:
                if f in vocab and doc[f]>0:
                    data_col.append(vocab[f])
                    data_row.append(count)
                    data.append(doc[f])
            count=count+1
    elif vocab==None:
        vocab={}
        generate_vocab() 
        for doc in dicts:
            for f in doc:
                if f in vocab and doc[f]>0:
                    data_col.append(vocab[f])
                    data_row.append(count)
                    data.append(doc[f])
            count=count+1
    X = csr_matrix((data, (data_row, data_col)),dtype=np.int64)
    return X,vocab


def accuracy_score(truth, predicted):
    return len(np.where(truth==predicted)[0]) / len(truth)


def cross_validation_accuracy(clf, X, labels, k):
    res = 0
    acc = []
    kf = KFold(len(labels),n_folds = k)
    for train, test in kf:
        x = labels[test]
        clf.fit(X[train], labels[train])
        y = clf.predict(X[test])
        acc.append(accuracy_score(x,y))
    return np.mean(acc)


def eval_all_combinations(docs, labels, punct_vals,
                          feature_fns, min_freqs):
    result=[]
    fts = defaultdict(lambda: 0)
    tokens_list = []
    list_accuracy = []
    combs = []
    for l in range(1,len(feature_fns)+1):
        for subset in combinations(feature_fns, l):
            combs.append(subset)
            
    tokens_false = [tokenize(d,False) for d in docs]
    tokens_true = [tokenize(d,True) for d in docs]
    
    for fns in combs:
        for freq in min_freqs:
            for pun in punct_vals:
                if pun == True:
                    X, vocab = vectorize(tokens_true,fns,freq)
                elif pun == False:
                    X, vocab = vectorize(tokens_false,fns,freq)
                acc = cross_validation_accuracy(LogisticRegression(), X, labels,k=5)
                result.append({"punct":pun,"features":fns,"min_freq":freq,"accuracy":acc})
                
    return(sorted(result,key = lambda x:-x["accuracy"]))

def plot_sorted_accuracies(results):
    res =sorted([acc["accuracy"] for acc in results])
    plt.plot(res)
    plt.ylabel('Accuracy')
    plt.savefig("accuracies.png")


def mean_accuracy_per_setting(results):
   res = []
    all_punct = []
    all_feat = []
    all_freq = []
    acc_punct = {}
    acc_freq = {}
    acc_feat = {}
    n = len(results)
    for data in results:
        if data["punct"] not in all_punct:
            all_punct.append(data["punct"])
        if data["features"] not in all_feat:
            all_feat.append(data["features"])
        if data["min_freq"] not in all_freq:
            all_freq.append(data["min_freq"])
    
    
    for data in results:
        if ("punct="+str(data["punct"])) not in acc_punct.keys():
            acc_punct[("punct="+str(data["punct"]))]=data["accuracy"]
        else:
            acc_punct[("punct="+str(data["punct"]))]+=data["accuracy"]
            
        if "features="+str(data["features"]) not in acc_feat.keys():
            acc_feat[("features="+str(data["features"]))]=data["accuracy"]
        else:
            acc_feat[("features="+str(data["features"]))]+=data["accuracy"]
            
        if "min_freq="+str(data["min_freq"]) not in acc_freq.keys():
            acc_freq[("min_freq="+str(data["min_freq"]))]=data["accuracy"]
        else:
            acc_freq[("min_freq="+str(data["min_freq"]))]+=data["accuracy"]

    for pun in acc_punct:
        res.append((acc_punct[pun]/(n/len(acc_punct)),pun))
    for ft in acc_feat:
        res.append((acc_feat[ft]/(n/len(acc_feat)),ft))
    for fr in acc_freq:
        res.append((acc_freq[fr]/(n/len(acc_freq)),fr))

    return sorted(res,key=lambda x:-x[0])
    


def fit_best_classifier(docs, labels, best_result):
    feature_fns = best_result["features"]
    token_list = [tokenize(d,best_result["punct"]) for d in docs]
    X, vocab = vectorize(token_list, best_result["features"],best_result["min_freq"])
    clf=LogisticRegression()
    clf.fit(X,labels)
    return clf,vocab


def top_coefs(clf, label, n, vocab):
    coef = clf.coef_[0]
    top_coef_terms = []
    top_coef = []
    if label == 1:
        top_coef_ind = np.argsort(coef)[::-1][:n]
    elif label == 0:
        top_coef_ind = np.argsort(coef)[::1][:n]

    t = sorted([(i,vocab[i]) for i in vocab], key = lambda x:x[0])
    
    
    for i in top_coef_ind:
        top_coef_terms.append(t[i])
        
    for i in range(len(top_coef_ind)):
        top_coef.append((top_coef_terms[i][0],abs(coef[top_coef_ind[i]])))
        
    return sorted(top_coef,key=lambda x:-x[1])[:n] 



def parse_test_data(best_result, vocab):
    docs, labels = read_data(os.path.join('data', 'test'))
    punct = best_result["punct"]
    token_list  = [tokenize(d,punct) for d in docs]
    feature_fns = best_result["features"]
    X_test, v = vectorize(token_list, feature_fns,best_result["min_freq"],vocab)

    return docs,labels,X_test

def print_top_misclassified(test_docs, test_labels, X_test, clf, n):
    predict = clf.predict(X_test)
    probab = clf.predict_proba(X_test)
    count = 0
    all_misclassified = []
    for i in range(len(test_docs)):
        if predict[i]!=test_labels[i]:
            all_misclassified.append({"truth":test_labels[i],"predicted":predict[i],"probability":probab[i][1],"docs":test_docs[i]})
            all_misclassified.append({"truth":test_labels[i],"predicted":predict[i],"probability":probab[i][0],"docs":test_docs[i]})
    
    all_misclassified = sorted(all_misclassified,key=lambda x:-x["probability"])[:n]
    
    for i in all_misclassified:
        print("truth=",i["truth"],"predicted=",i["predicted"],"probability=",i["probability"])
        print(i["docs"])
        print("\n")


def main():
    feature_fns = [token_features, token_pair_features, lexicon_features]
    # Download and read data.
    download_data()
    docs, labels = read_data(os.path.join('data', 'train'))
    # Evaluate accuracy of many combinations
    # of tokenization/featurization.
    results = eval_all_combinations(docs, labels,
                                    [True, False],
                                    feature_fns,
                                    [2,5,10])
    # Print information about these results.
    best_result = results[0]
    worst_result = results[-1]
    print('best cross-validation result:\n%s' % str(best_result))
    print('worst cross-validation result:\n%s' % str(worst_result))
    plot_sorted_accuracies(results)
    print('\nMean Accuracies per Setting:')
    print('\n'.join(['%s: %.5f' % (s,v) for v,s in mean_accuracy_per_setting(results)]))

    # Fit best classifier.
    clf, vocab = fit_best_classifier(docs, labels, results[0])

    # Print top coefficients per class.
    print('\nTOP COEFFICIENTS PER CLASS:')
    print('negative words:')
    print('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 0, 5, vocab)]))
    print('\npositive words:')
    print('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 1, 5, vocab)]))

    # Parse test data
    test_docs, test_labels, X_test = parse_test_data(best_result, vocab)

    # Evaluate on test set.
    predictions = clf.predict(X_test)
    print('testing accuracy=%f' %
          accuracy_score(test_labels, predictions))

    print('\nTOP MISCLASSIFIED TEST DOCUMENTS:')
    print_top_misclassified(test_docs, test_labels, X_test, clf, 5)


if __name__ == '__main__':
    main()
