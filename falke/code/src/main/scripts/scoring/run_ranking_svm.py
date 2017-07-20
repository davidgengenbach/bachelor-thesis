
'''
train a ranking svm and apply to given data
'''
import sys
sys.path.insert(0, '../pyutil')
import weka
import dlib
import pickle

# params
mode = sys.argv[1]
assert mode in ['train','rank']
file = sys.argv[2]

C = 30

# load data
print file
data = weka.arff_to_list(file)
print len(data), len(data[0])

# expected weka format: 0 -> id, 1 -> topic, attributes, last -> binary label
first_att = 2
last_att = len(data[0])-2
label = len(data[0])-1


# train
if mode == 'train':
    print 'training'

    # create training dataset
    t2p = {}
    for row in data:
        if row[1] not in t2p:
            t2p[row[1]] = dlib.ranking_pair()
        f = dlib.vector(row[first_att:last_att+1])
        if row[label] == 0:
            t2p[row[1]].nonrelevant.append(f)
        else:
            t2p[row[1]].relevant.append(f)
    
    topics = dlib.ranking_pairs()
    for t in t2p:
        topics.append(t2p[t])
    print 'topics', len(topics)
    
    # train
    trainer = dlib.svm_rank_trainer()
    trainer.c = C
    rank = trainer.train(topics)
    print dlib.test_ranking_function(rank, topics)
    
    with open(file.replace('arff','pkl'), 'wb') as f:
        pickle.dump(rank, f)
    print 'model saved to', file.replace('arff','pkl')

    #print("Cross validation results: {}".format(
    #    dlib.cross_validate_ranking_trainer(trainer, queries, 4)))

# rank
if mode == 'rank':
    print 'ranking'
    
    model_file = sys.argv[3]
    with open(model_file+'.pkl', 'rb') as f:
        rank = pickle.load(f)
    
    with open(file.replace('arff','ranked'), 'w') as out:
        for row in data:
            f = dlib.vector(row[first_att:last_att+1])
            score = rank(f)
            out.write('{:.0f}\t{:.0f}\t{}\n'.format(row[0],row[1],score))
    
