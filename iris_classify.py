from mrjob.job import MRJob
from mrjob.step import MRStep
import heapq
from operator import itemgetter
import pandas as pd
import math

f=open(r"E:\dcstorage_proj\task3\Iris.csv")
dataf=pd.read_csv(f)     #Open Iris file, save the data into the dataframe called dataf
speciesNaN = dataf.loc[dataf['Species'].isnull()]  #Find species-unknown points,saved as speciesNaN


class IrisClassify(MRJob):
    #The methods below will be executed as the order displayed.

    #This init method saves all the speciesNaN points, each respectively loaded into list,
    # also saves the max&min value under four features respectively
    def mapper_init_spNAN_featureMAXMIN(self):
        self.unknown = speciesNaN.values.tolist()  # Load speciesNaN into list
        # max&min of feature1
        self.SpLmax=dataf.loc[:,"SepalLengthCm"].max()
        self.SpLmin=dataf.loc[:,"SepalLengthCm"].min()
        # max&min of feature2
        self.SpWmax=dataf.loc[:,"SepalWidthCm"].max()
        self.SpWmin=dataf.loc[:,"SepalWidthCm"].min()
        # max&min of feature3
        self.PtLmax=dataf.loc[:,"PetalLengthCm"].max()
        self.PtLmin = dataf.loc[:, "PetalLengthCm"].min()
        # max&min of feature4
        self.PtWmax = dataf.loc[:, "PetalWidthCm"].max()
        self.PtWmin = dataf.loc[:, "PetalWidthCm"].min()

        # We use iterative method to normalize features' values of the three SpeciesNaN points
        for i in range(0,len(self.unknown)):
            self.unknown[i][1] = (self.unknown[i][1] - self.SpLmin) / (self.SpLmax - self.SpLmin)
            self.unknown[i][1] = float('%.4f' % (self.unknown[i][1]))  # Keep the format of four decimal places and transfer to float type
            self.unknown[i][2] = (self.unknown[i][2] - self.SpWmin) / (self.SpWmax - self.SpWmin)
            self.unknown[i][2] = float('%.4f' % self.unknown[i][2])
            self.unknown[i][3] = (self.unknown[i][3] - self.PtLmin) / (self.PtLmax - self.PtLmin)
            self.unknown[i][3] = float('%.4f' % self.unknown[i][3])
            self.unknown[i][4] = (self.unknown[i][4] - self.PtWmin) / (self.PtWmax - self.PtWmin)
            self.unknown[i][4] = float('%.4f' % self.unknown[i][4])

    # The mapper yields ((Id,features1~4),species) pairs for those with known species
    def mapper_knownspecies(self, _, line):
        line=line.split(',')

        if (line[0]!='Id'and line[5]!=''):#Skip the heading and the SpeciesNaN lines
            # Transfer each feature's value into float type
            line[1]=float(line[1])
            line[2]=float(line[2])
            line[3]=float(line[3])
            line[4]=float(line[4])
            yield line[0:5],(line[5])
            #yield ((Id,feature1,..feature4),species) pair for those with known species

   #From the ((Id,feature1,..feature4),species) pair got above, normalize each feature in the (Id, feature1..feature4)
    def mapper_normalize(self,IdFeatures,species):   #The formula of normalization referenced [5]
        #Normalize feature1
        IdFeatures[1]=(IdFeatures[1]-self.SpLmin)/(self.SpLmax-self.SpLmin)
        IdFeatures[1]=float('%.4f' % IdFeatures[1])   #Keep mostly 4 decimal places
        # Normalize feature2
        IdFeatures[2] = (IdFeatures[2]-self.SpWmin)/(self.SpWmax-self.SpWmin)
        IdFeatures[2]=float('%.4f' %IdFeatures[2])
        # Normalize feature3
        IdFeatures[3]=(IdFeatures[3]-self.PtLmin)/(self.PtLmax-self.PtLmin)
        IdFeatures[3] = float('%.4f' %IdFeatures[3])
        # Normalize feature4
        IdFeatures[4]=(IdFeatures[4]-self.PtWmin)/(self.PtWmax-self.PtWmin)
        IdFeatures[4] = float('%.4f' %IdFeatures[4])
        yield IdFeatures[0:5],species
        # yield ((Id,feature1,..feature4),species) pair for those with known species,but now features
           # already normalized


    #This mapper mainly calculates the Euclidean distances between each SpeciesNaN point and each species-known point
    #Input is above output : ((Id,feature1,..feature4),species) pair with features nermalized
    #Output: (SpecisNaN point's ID, (Euclidean distance with a species-known point, known's species) pair
    def mapper_unknown_distances(self,ndt,species):

        # Iterate over SpeciesNaN data points
        for i in range(0,len(self.unknown)):
            # Squares of difference between each speciesNaN
            #  point and each inputted species-known point
            squares = [0, 0, 0]
            # Distances between each speciesNaN point and inputted species-known point,
            #  namely sqrt(squares)
            dists = [0, 0, 0]
            # Squares and distances here should set as 0s initially and after a computation

            # Iterate over features, then calculate Euclidean
            #  distance through each feature
            for j in range(1,5):
                squares[i]+=(self.unknown[i][j]-ndt[j])**2
            dists[i]=math.sqrt(squares[i])

            #yield (SpecisNaN point's ID, (Euclidean distance with a species-known point, known's species) pair
            yield (self.unknown[i][0],), (dists[i],species)



    #This reducer groups the distance & species data by the SpecisNaN point's ID, and displays
    # the species-known points whose distance is in the 15 smallest ones
    #Input: Above output: (speciesNaN point's ID,(distance with a species-known,its species)) pair
    #Output: (speciesNaN ID, collection of 15 distance_species of known)
    def reducer_byunknwonID(self,id, dist_species):

        yield id,heapq.nsmallest(15, dist_species, key=itemgetter(0))  #nsmallest regards first dimension distance as key


    #This mapper mainly maps species to (species,1) in the collection under the speciesNaN ID
    #Input: above output: (speciesNaN ID, collection of 15 distance_species of known)
    #Output:((speciesNaN ID, each species in the collection), 1)
    def mapper_unID_species(self,id,dist_species):
        for item in dist_species:

            yield (id, item[1]),1


    #This combiner sums the 1s got above for a species under the speciesNaN point
    #Input: above output: ((speciesNaN ID, each species in the collection), 1) pair
    #Output:(speciesNaN id, (total occurrences of a species-known's species, the species)) pair
    def combiner_unknown_a_species(self,id_species,count):
        yield id_species[0],(sum(count),id_species[1])


    #This reducer mainly figures out the species which appears the most
    #Input: above output (speciesNaN id, (total occurrences of a species-known's species, the species)) pair
    #Output: (speciesNaN id, (occurrence of a species, the species)) in which species appeared mostly
    def reducer_unknown_classify(self,id,sums_species):
        yield id,max(sums_species)      #max defaultly regards the first dimension occurrence as key


    #This mapper just derives ((speciesNaN id, species) from the above result
    #Input: (speciesNaN id, (occurrence of a species, the species)) in which species appeared mostly
    #Output: (speciesNaN id, predicted species) pair
    def mapper_classify(self,id,sum_species):
        yield id,sum_species[1]



    def steps(self):

        return [
            MRStep(mapper_init=self.mapper_init_spNAN_featureMAXMIN
                   ,mapper=self.mapper_knownspecies),
            MRStep(mapper_init=self.mapper_init_spNAN_featureMAXMIN,
                  mapper=self.mapper_normalize),
            MRStep(mapper_init=self.mapper_init_spNAN_featureMAXMIN,
                   mapper=self.mapper_unknown_distances,
                   reducer=self.reducer_byunknwonID),
            MRStep(mapper=self.mapper_unID_species,
                   combiner=self.combiner_unknown_a_species,
                    reducer=self.reducer_unknown_classify),
            MRStep(mapper=self.mapper_classify)

            ]





if __name__ == '__main__':
    IrisClassify.run()
