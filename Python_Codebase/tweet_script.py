import tweepy

ckey = ""
csecret = ""
atoken = ""
asecret = ""

OAUTH_KEYS = {'consumer_key':ckey, 'consumer_secret':csecret,\
    'access_token_key':atoken, 'access_token_secret':asecret}
auth = tweepy.OAuthHandler(OAUTH_KEYS['consumer_key'], OAUTH_KEYS['consumer_secret'])
api = tweepy.API(auth)

cricTweet = tweepy.Cursor(api.search, q='cricket', \
    geocode="-125.0011, 24.9493, -66.9326, 49.5904,-179.1506, 51.2097, \
    -129.9795, 71.4410, -160.2471, 18.9117, -154.8066, 22.2356").items(10)

for tweet in cricTweet:
    print(tweet)
