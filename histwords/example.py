from representations.sequentialembedding import SequentialEmbedding
import collections

"""
Example showing how to load a series of historical embeddings and compute similarities over time.
Warning that loading all the embeddings into main memory can take a lot of RAM
"""

def get_seq_closest(fiction_embeddings, word, start_year, num_years=10, n=10):
    closest = collections.defaultdict(float)
    for year in range(start_year, start_year + num_years, 10):
        #print("\nseq_closest_" +  str(start_year) + ":")
        embed = fiction_embeddings.embeds[year]
        year_closest = embed.closest(word, n=n*10)
        #print(year_closest)
        #print()
        for score, neigh in year_closest:
            closest[neigh] += score
        
    return sorted(closest, key = lambda word : closest[word], reverse=True)[0:n]

if __name__ == "__main__":
    print('Running: ...')
 
    fiction_embeddings = SequentialEmbedding.load("embeddings/eng-fiction-all_sgns", range(1900, 2000, 10))
    
    pairs = [["lesbian", "gay"], ["engineer", "man"], ["nurse", "woman"], ["labor", "man"], ["math", "man"], ["arts", "woman"], ["police", "woman"], ["military", "woman"]]
    #time_sims = fiction_embeddings.get_time_sims("lesbian", "gay")   
    
    for i in range(0, len(pairs)):
        time_sims = fiction_embeddings.get_time_sims(pairs[i][0], pairs[i][1])
        print "Similarity between " + pairs[i][0] + " and " + pairs[i][1] + " from 1900s to the 1990s:"
        for year, sim in time_sims.iteritems():
            print "{year:d}, cosine similarity={sim:0.2f}".format(year=year,sim=sim)
        print("\n")
    
    #neighbour = fiction_embeddings.get_seq_neighbour_set("lesbian")    
    #print("Neighbour: ", len(neighbour), neighbour)

    
    #feminine_gender = ['feminine', 'lady', 'woman', 'girl']
    feminine_gender = ['teacher', 'nurse', 'caretaker', 'cleaner', 'lesbian']
    for word in feminine_gender:
        print('Word: ', word)
        for year in range(10, 100, 10):
            #closest = get_seq_closest(fiction_embeddings, word, 1900, year)
            closest = get_seq_closest(fiction_embeddings, word, 1900+year, 10)
            print("\tClosest   ", closest)

    print("\n")
    #masculine_gender = ['masculine', 'gentleman', 'man', 'boy']
    masculine_gender = ['engineer', 'doctor', 'electrician', 'plumber', 'gay']
    for word in masculine_gender:
        print('Word: ', word)
        for year in range(10, 100, 10):
            #closest = get_seq_closest(fiction_embeddings, word, 1900, year)
            closest = get_seq_closest(fiction_embeddings, word, 1900+year, 10)
            print("\tClosest   ", closest)
    

