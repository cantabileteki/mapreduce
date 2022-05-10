from mrjob.job import MRJob
from mrjob.step import MRStep

import math

class FrobeniusNorm(MRJob):
    #The methods below will be executed as the order displayed.

    #This mapper splits the line into a list with 50 numbers, and we will do calculation
    # on the list in the next reducer.
    #Input:None, each line in txt
    #Output:(None, a list with 50 numbers)
    def mapper_split(self,_,line):
        line=line.split()

        for i in range(len(line)):
            line[i]=float(line[i])

        yield None,line

    #This reducer calculates the sum of squares over a row
    #Input:above output: (None, a list with 50 numbers) pair
    #Output: (None, sum of squares over the list) pair
    def reducer_row_squaresum(self,_,row):


        for numberL in row:   #Iterate the lists over the row(value), actually just one list in the row
            squaresum = 0    #The sum of squares should be 0 initially and after a computation over the list(row)
            for number in numberL:     #Iterate the numbers in the list
                squaresum+=number**2    #sum of squares

            yield None,squaresum

    #This reducer calculates the norm, namely the square root of squares'sum over all rows
    #Input: above output:(None, sum of the squares over one row)
    #Output:(square root of sum of squares over all rows, None) pair
    def reducer_norm(self,_,squaresum):

        result=math.sqrt(sum(squaresum))
        yield result,None





    def steps(self):

        return [
            MRStep(mapper=self.mapper_split,
                   reducer=self.reducer_row_squaresum),
            MRStep(reducer=self.reducer_norm)


            ]





if __name__ == '__main__':
    FrobeniusNorm.run()
