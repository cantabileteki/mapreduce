
from mrjob.job import MRJob
from mrjob.step import MRStep
import nltk
from operator import itemgetter

import heapq
import pandas as pd
import re
from nltk.corpus import stopwords

class ToptenWords(MRJob):
    #The methods below will be executed as the order displayed.

    # This mapper yields (title, genres) pair for each movie:
    def mapper_title_genres(self, _, line):  # Referenced [1]
        # By observation，we found some titles are only with spaces, like "Toy Story",
        #  so it's fine to split the corresponding line just with comma thus we can get the full title.
        # While other titles are with punctuations, like
        #  "\"The Death of "Superman Lives": What Happened? (2015) \", we should split the corresponding line at the first and
        # the last comma, for which we use regularization expression below to split
        if re.match(r'\d+', line) is not None:  # skip heading，which doesn't involve any numbers
            if '"' in line:  # when title with punctuations, namely which in \"...\"
                PATTERN = r',(.+),'
                title = str(re.findall(PATTERN, line))  # find title between the first and last comma
                line = re.split(PATTERN, line)  # same pattern split the line

                yield title, line[-1]  # yield title,genres

            else:
                line = line.split(',')  # when title only with spaces
                yield line[1], line[2]  # yield title，genres

    #This mapper yields (title, [genre1,genre2,genre3...]) pair for above output (title, genres)
    #Namely mainly to list all genres for a movie
    def mapper_split_genres(self,title,genres):
        genres_split = genres.split('|')#split at "|"，then all possible genres of a movie loaded in a list
        yield title,genres_split  #yield title, [genre1,genre2,genre3...]


    # According to above output (title, [genre1,genre2,genre3...]), yield
    # (genre1, title),(genre2,title )...pairs for each movie
    def mapper_genre_title(self, title, genrelist):
        for each_genre in genrelist:
            yield  each_genre,title


    # Process the title in above output (genre1, title) with nltk library,
    #  yield （genre, [title with keywords]）pair
    def mapper_process_titles(self,genre,title):
        title_keep_keywords=filter_title(title)   #Filter out punctuations，numbers，conjunctions，etc... from title
        yield genre,title_keep_keywords         # and tokenize it. Definition of filter_title function shown at the end

    #According to above output （genre, [title with keywords]）pair ,for
    # each keyword in [title with keywords], we yield ((genre,word),1) pair
    def mapper_with_genre(self,genre,title_kwds):
        for kword in title_kwds:
            yield (genre+":",kword),1  #for instance, yield (Action:, Man),1 pair

    #For above output ((genre,word),1) pair, sum ‘1’s based on its (genre, word)，which generates ((genre, word), the
    # word's occurrence under the genre) pair. And this combiner's computation runs at each node next to the mapper
    def combiner_by_gen_kword(self,gen_word,count):
        yield gen_word,sum(count)  #for instance, yield (Action:, Man),3 pair

    #This reducer does the same task as above combiner, namely according to above output
    # ((genre, word), word's occurrence) we sum occurrence, but this time getting a word's
    # total occurrence under the genre
    def reducer_by_gen_kword(self,gen_word,counts):
        yield gen_word[0],(gen_word[1],sum(counts))   #for instance, yield (Action:,( Man,10)) pair

    #This reducer lists 10 appeared-mostly words under each genre with the word's occurrence
    #Acoording to above output ((genre, (word, word's total occurrence)) pair, we yield
    # (genre, ((word1, its occurrence),(word2, its occurrence)...) in which the words are 10 appeared-mostly)
    def reducer_top10_kword(self,genre,wordsum):
        yield genre, heapq.nlargest(10,wordsum,key=itemgetter(1))     #Referenced [2]


    #This mapper lists 10 appeared-mostly words under each genre but without the word's occurrence
    #According to above output (genre, ((word1, its occurrence),(word2, its occurrence)...),
    # we yield (genre, (word1,word2...))  in which the words are 10 appeared-mostly)
    def mapper_top10word(self,genre,wordlist):
        top10wd_genre=[]
        for wordsum in wordlist:
            top10wd_genre.append(wordsum[0])   #derive words from value(wordlist) and loaded into list
        yield genre,top10wd_genre



    def steps(self):

        return [
            MRStep(mapper=self.mapper_title_genres),
            MRStep(mapper=self.mapper_split_genres),
            MRStep(mapper=self.mapper_genre_title),
            MRStep(mapper= self.mapper_process_titles),
            MRStep(mapper=self.mapper_with_genre,
                   combiner=self.combiner_by_gen_kword,
                   reducer=self.reducer_by_gen_kword),
            MRStep(reducer=self.reducer_top10_kword),
            MRStep(mapper=self.mapper_top10word)

            ]

#filter_title function is used to filter out punctuations，numbers，conjunctions，etc... from title
# and tokenize it
# Below packet involving English and Chinese punctuations and numbers, referenced [3]
extrapuncs='[0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~‘’“”]+'
def filter_title(moviestr):
    filtered_title = []  # This list is for the tokens after filtering
    moviestr=re.sub(extrapuncs, '', moviestr)  #filter out punctuations, numbers
    mvtxtL = nltk.word_tokenize(moviestr)  # tokenize, referenced [4]

    for i in range(0,len(mvtxtL)):
        mvtxtL[i]=mvtxtL[i].lower()   #every word should be lower case

    #Remove the words in English stopwords and in French stopwords
    for word in mvtxtL:
        if (word not in stopwords.words('english')) and (word not in stopwords.words('french')):
            filtered_title.append(word)
    return filtered_title  #Get a filtered title, which is a tokens' list



if __name__ == '__main__':

    ToptenWords.run()

