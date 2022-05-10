from mrjob.job import MRJob
from mrjob.step import MRStep

class webgraph(MRJob):
    #The methods below will be executed as the order displayed.

    # We yield (To NodeId, From Node Id) pair, namely reverse the direction for each edge
    def mapper_target_source(self, _, line):
        line=line.strip().split()
        if line[0]!='#':  #Skip headings
            yield line[1],line[0]  #yield (To NodeId, From Node Id) pair

    # From above output (To NodeId, From Node Id) pair, this combiner yields (target, a list of sources who have same target) pair
    # But here the sources haven't been completely grouped by since the computation runs at each node next to the mapper
    def combiner_bytarget(self,target,source):
        yield target,[x for x in source]

    #This reducer does the same task as above combiner, from above output (target, a list of partial sources)
    # we yield (target, a list of total sources)
    def reducer_bytarget(self,target,sourceL):
        sourceL_bytarget=[]
        for source in sourceL:  #Iterate the lists over sourceL(values got above), actually just one list
            for x in source:    #Iterate each node in the list
                sourceL_bytarget.append(x) #The total sources under a target loaded in a list
        yield target,sourceL_bytarget



    def steps(self):

        return [
            MRStep(mapper=self.mapper_target_source,
                   combiner=self.combiner_bytarget,
                    reducer=self.reducer_bytarget)

            ]


if __name__ == '__main__':
    webgraph.run()
