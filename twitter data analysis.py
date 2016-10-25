from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
import sys
import time
from TwitterAPI import TwitterAPI

consumer_key = 'Input yours'
consumer_secret = 'Input yours'
access_token = 'Input yours'
access_token_secret = 'Input yours'

def get_twitter():
    return TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)


def read_screen_names(filename):
    candidate = []
    with open(filename, "r") as ins:        
        for line in ins:
            candidate.append(line.rstrip())
    return candidate

def robust_request(twitter, resource, params, max_tries=5):
    for i in range(max_tries):
        request = twitter.request(resource, params)
        if request.status_code == 200:
            return request
        else:
            print('Got error %s \nsleeping for 15 minutes.' % request.text)
            sys.stderr.flush()
            time.sleep(61 * 15)
            
            
def get_users(twitter,screen_names):
    return robust_request(twitter, 'users/lookup', {'screen_name':screen_names}, max_tries=5).json()
    #return twitter.request('users/lookup',{'screen_name':screen_names}).json()
     

def get_friends(twitter, screen_name):
    return sorted(((robust_request(twitter, 'friends/ids', {'screen_name':screen_name,'count':5000}, max_tries=5)).json())['ids'])
  
def add_all_friends(twitter, users):
    friends = []
    for i in range(len(users)):
        friendlist=get_friends(twitter, users[i]['screen_name'])
        users[i]['friends'] = friendlist
            
def print_num_friends(users):
    count=0
    for user in users:
        print(user['screen_name']+" "+str(user['friends_count']))

def count_friends(users):
    cnt = Counter()
    all_friends = []
    for i in range(len(users)):
        all_friends=all_friends+users[i]['friends']
    cnt = Counter(all_friends)
    return cnt

    
def friend_overlap(users):
	x=[]
    a=[set(users[i]['friends']) for i in range(len(users))]
    for i in range(len(users)-1):
        for j in range(i+1,len(users)):
            x=x+[(users[i]['screen_name'],users[j]['screen_name'],len(a[i]&a[j]))]
            x=sorted(x,key=lambda x:-x[2])
    return x


def followed_by_hillary_and_donald(users, twitter):
	a=[]
    for i in range(len(users)):
        if users[i]['screen_name'] == 'realDonaldTrump' or users[i]['screen_name']=='HillaryClinton':
            a.append(users[i]['friends'])
    #generalised if more names are added to if codition
    id= set(a[0])
    for i in range(1,len(a)):
        id = id&set(a[i])
        
    #followedby = twitter.request('users/lookup',{'user_id':id}).json()
    if id == []:
    	followedby = (robust_request(twitter, 'users/lookup', {'user_id':id}, max_tries=5)).json()
    else:
    	followedby = []
    return [followedby[i]['screen_name'] for i in range(len(followedby))]

def create_graph(users, friend_counts):
    G=nx.Graph()
    
    for user in users:
        G.add_node(user['screen_name'])
                
    for i in friend_counts:
        if friend_counts[i]>1:
            G.add_node(i)
            for j in users:
                if i in j["friends"]:
                    G.add_edge(i,j['screen_name'])              
    return G

def draw_network(graph, users, filename):
    user_labels = {}
    for user in users:
        user_labels[user["screen_name"]] = user["screen_name"]
    nx.draw_networkx(graph, labels=user_labels, node_size=10, width=.2, edge_color='grey', alpha=.5)
    plt.axis('off')
    plt.savefig(filename)
    
    
def main():
    twitter = get_twitter()
    screen_names = read_screen_names('candidates.txt')
    print('Established Twitter connection.')
    print('Read screen names: %s' % screen_names)
    users = sorted(get_users(twitter, screen_names), key=lambda x: x['screen_name'])
    print('found %d users with screen_names %s' %
          (len(users), str([u['screen_name'] for u in users])))
    add_all_friends(twitter, users)
    print('Friends per candidate:')
    print_num_friends(users)
    friend_counts = count_friends(users)
    print('Most common friends:\n%s' % str(friend_counts.most_common(5)))
    print('Friend Overlap:\n%s' % str(friend_overlap(users)))
    print('User followed by Hillary and Donald: %s' % followed_by_hillary_and_donald(users, twitter))

    graph = create_graph(users, friend_counts)
    print('graph has %s nodes and %s edges' % (len(graph.nodes()), len(graph.edges())))
    draw_network(graph, users, 'network.png')
    print('network drawn to network.png')


if __name__ == '__main__':
    main()
