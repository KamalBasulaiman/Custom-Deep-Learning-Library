import numpy as np
from math import *
from numpy import *

'''
SymbolSets: A list containing all the symbols (the vocabulary without blank)

y_probs: Numpy array with shape (# of symbols + 1, Seq_length, batch_size)
         Your batch size for part 1 will remain 1, but if you plan to use your
         implementation for part 2 you need to incorporate batch_size.

Return the forward probability of the greedy path (a float) and
the corresponding compressed symbol sequence i.e. without blanks
or repeated symbols (a string).
'''
def GreedySearch(SymbolSets, y_probs):

    final_probablity = 1.0;     seq_path = [];    txt = '';     seq_len = np.shape(y_probs)[1]
    for i in range(seq_len):    
        k = np.argmax(y_probs[:,i,:])
        if k == 0:  seq_path.append("###")
        elif k !=0: 
            seq_path.append(SymbolSets[k-1])
            if i == 0 or seq_path[i] != seq_path[i-1]: 
                txt = txt + SymbolSets[k-1]
        final_probablity *= y_probs[k,i,:]  
    return (txt, final_probablity)



##############################################################################



'''
SymbolSets: A list containing all the symbols (the vocabulary without blank)

y_probs: Numpy array with shape (# of symbols + 1, Seq_length, batch_size)
         Your batch size for part 1 will remain 1, but if you plan to use your
         implementation for part 2 you need to incorporate batch_size.

BeamWidth: Width of the beam.

The function should return the symbol sequence with the best path score
(forward probability) and a dictionary of all the final merged paths with
their scores.
'''
def BeamSearch(SymbolSets, y_probs, BeamWidth):
    
    BlankPathScore = {}
    PathScore = {}
    
    NewPathsWithTerminalBlank, NewPathsWithTerminalSymbol, NewBlankPathScore, NewPathScore = InitializePaths(SymbolSets, y_probs[:,0,0], BlankPathScore, PathScore, BeamWidth)
    seq_len = np.shape(y_probs)[1]
    for t in range(1,seq_len):
        PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore = Prune(NewPathsWithTerminalBlank, NewPathsWithTerminalSymbol, NewBlankPathScore, NewPathScore, BeamWidth)

        NewPathsWithTerminalBlank, NewBlankPathScore = ExtendWithBlank(PathsWithTerminalBlank, PathsWithTerminalSymbol, y_probs[:,t,0],BlankPathScore, PathScore)

        NewPathsWithTerminalSymbol, NewPathScore = ExtendWithSymbol(PathsWithTerminalBlank, PathsWithTerminalSymbol, SymbolSets, y_probs[:,t,0],BlankPathScore, PathScore)
        
    MergedPaths, FinalPathScore = MergeIdenticalPaths(NewPathsWithTerminalBlank, NewPathsWithTerminalSymbol, NewPathScore, NewBlankPathScore)

    probs = FinalPathScore.values()
    max_prob = np.max(np.array(sorted(list(probs),reverse=True)))
    keys = FinalPathScore.keys()
    seq = 'sequance'
    length = len(FinalPathScore)
    i = 0
    while(i < length):
        if FinalPathScore[list(keys)[i]] == max_prob:
            seq = list(keys)[i]
        i += 1

    return (str(seq), FinalPathScore)

def InitializePaths(SymbolSet, y, BlankPathScore, PathScore, BeamWidth):

    InitialBlankPathScore = {}
    path = '' 
    InitialPathScore = {}
    InitialBlankPathScore[path] = y[0]
    InitialPathsWithFinalBlank = {path}
    InitialPathsWithFinalSymbol = set()
    k = 1
    for c in SymbolSet:
        path = c
        InitialPathScore[path] = y[k]
        InitialPathsWithFinalSymbol.add(path)
        k += 1
    return InitialPathsWithFinalBlank, InitialPathsWithFinalSymbol, InitialBlankPathScore, InitialPathScore

def ExtendWithBlank(PathsWithTerminalBlank, PathsWithTerminalSymbol, y, BlankPathScore, PathScore):
    UpdatedPathsWithTerminalBlank = set()
    UpdatedBlankPathScore = {} 

    for path in PathsWithTerminalBlank:
        # Set addition
        UpdatedPathsWithTerminalBlank.add(path) 
        UpdatedBlankPathScore[path] = BlankPathScore[path] * y[0]

    for path in PathsWithTerminalSymbol:
        if path in UpdatedPathsWithTerminalBlank:   UpdatedBlankPathScore[path] += PathScore[path]* y[0]
        else:
            # Set addition
            UpdatedPathsWithTerminalBlank.add(path) 
            UpdatedBlankPathScore[path] = PathScore[path] * y[0]

    return UpdatedPathsWithTerminalBlank, UpdatedBlankPathScore

def Prune(PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore, BeamWidth):
    scorelist = []
    PrunedBlankPathScore = {}
    PrunedPathScore = {}

    for p in PathsWithTerminalBlank:
        scorelist.append(BlankPathScore[p])

    for p in PathsWithTerminalSymbol:
        scorelist.append(PathScore[p])

    scorelist.sort(reverse = True)    

    if BeamWidth >= len(scorelist):     cutoff = scorelist[-1]
    else:                               cutoff = scorelist[BeamWidth]

    PrunedPathsWithTerminalBlank = set()
    for p in PathsWithTerminalBlank:
        
        if BlankPathScore[p] > cutoff:
            PrunedPathsWithTerminalBlank.add(p)
            PrunedBlankPathScore[p] = BlankPathScore[p]

    PrunedPathsWithTerminalSymbol = set() 
    for p in PathsWithTerminalSymbol:
        
        if PathScore[p] > cutoff:
            PrunedPathsWithTerminalSymbol.add(p)
            PrunedPathScore[p] = PathScore[p]

    return PrunedPathsWithTerminalBlank, PrunedPathsWithTerminalSymbol, PrunedBlankPathScore, PrunedPathScore

def MergeIdenticalPaths(PathsWithTerminalBlank, PathsWithTerminalSymbol, PathScore, BlankPathScore):
    MergedPaths = PathsWithTerminalSymbol
    FinalPathScore = PathScore
    for p in PathsWithTerminalBlank:
        if p in MergedPaths:    FinalPathScore[p] += BlankPathScore[p]
        else:
            # Set addition
            MergedPaths.add(p)
            FinalPathScore[p] = BlankPathScore[p]

    return MergedPaths, FinalPathScore

def ExtendWithSymbol(PathsWithTerminalBlank, PathsWithTerminalSymbol, SymbolSet, y, BlankPathScore, PathScore):
    UpdatedPathsWithTerminalSymbol = set()  
    UpdatedPathScore = {}
    
    for path in PathsWithTerminalBlank:
        k = 1
        for c in SymbolSet: # SymbolSet does not include blanks
            newpath = path + c # Concatenation
            UpdatedPathsWithTerminalSymbol.add(newpath) # Set addition
            UpdatedPathScore[newpath] = BlankPathScore[path] * y[k]
            k+=1

    for path in PathsWithTerminalSymbol:
        k = 1
        for c in SymbolSet: # SymbolSet does not include blanks
            if c == path[-1]:       newpath = path
            elif c != path[-1]:     newpath = path + c
            if newpath in UpdatedPathsWithTerminalSymbol:       UpdatedPathScore[newpath] += PathScore[path] * y[k] 
            else:
                UpdatedPathsWithTerminalSymbol.add(newpath) # Set addition
                UpdatedPathScore[newpath] = PathScore[path] * y[k] 
            k+=1

    return UpdatedPathsWithTerminalSymbol, UpdatedPathScore