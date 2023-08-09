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

    forward_path = []
    forward_prob = 1.0
    string = ''
    for i in range(np.shape(y_probs)[1]):
        index = 0
        for j in range(np.shape(y_probs)[0]):
            if j == 0:
                forward_path.append(y_probs[j,i,:])
                index = j
            else:
                if forward_path[len(forward_path)-1] < y_probs[j,i,:]:
                    forward_path[i] = y_probs[j,i,:]
                    index = j

        if index == 0:
            forward_path[i] = "_"
        else:
            forward_path[i] = SymbolSets[index-1]

        if index != 0: 
            if i == 0: 
                string += SymbolSets[index-1]
            elif forward_path[i-1] != forward_path[i]:
                string += SymbolSets[index-1]

        forward_prob = forward_prob * y_probs[index,i,:]
    forward_path = string

    return (forward_path, forward_prob)



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
    # return (bestPath, mergedPathScores)


    PathScore = {}
    BlankPathScore = {}
    NewPathsWithTerminalBlank, NewPathsWithTerminalSymbol, NewBlankPathScore, NewPathScore = InitializePaths(SymbolSets, \
    y_probs[:,0,0], BlankPathScore, PathScore, BeamWidth)

    # Subsequent time steps
    for t in range(1,np.shape(y_probs)[1]):
        PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore = Prune(NewPathsWithTerminalBlank, \
        NewPathsWithTerminalSymbol, NewBlankPathScore, NewPathScore, BeamWidth)
        # First extend paths by a blank
        NewPathsWithTerminalBlank, NewBlankPathScore = ExtendWithBlank(PathsWithTerminalBlank, PathsWithTerminalSymbol, \
        y_probs[:,t,0],BlankPathScore, PathScore)

        # Next extend paths by a symbol
        NewPathsWithTerminalSymbol, NewPathScore = ExtendWithSymbol(PathsWithTerminalBlank, PathsWithTerminalSymbol, \
        SymbolSets, y_probs[:,t,0],BlankPathScore, PathScore)
        # Prune the collection down to the BeamWidth
        # prune
        

    # merge identical paths
    MergedPaths, FinalPathScore = MergeIdenticalPaths(NewPathsWithTerminalBlank, NewPathsWithTerminalSymbol, NewPathScore, NewBlankPathScore)


    scores = sorted(list(FinalPathScore.values()),reverse=True)
    keys = list(FinalPathScore.keys())
    bestPath = ""
 
    for i in range(len(FinalPathScore)):
        if FinalPathScore[keys[i]] == scores[0]:
            bestPath = keys[i]

    return (bestPath, FinalPathScore)

def InitializePaths(SymbolSet, y, BlankPathScore, PathScore, BeamWidth):
    # First push the blank into a path-ending-with-blank stack. No symbol has been invoked yet
    InitialBlankPathScore = {}
    InitialPathScore = {}
    path = "" 
    InitialBlankPathScore[path] = y[0] # Score of blank at t=1
    InitialPathsWithFinalBlank = {path}
    # Push rest of the symbols into a path-ending-with-symbol stack
    InitialPathsWithFinalSymbol = set()
    index = 1
    for c in SymbolSet: # This is the entire symbol set, without the blank
        path = c
        InitialPathScore[path] = y[index]
        InitialPathsWithFinalSymbol.add(path)
        index += 1
    return InitialPathsWithFinalBlank, InitialPathsWithFinalSymbol, InitialBlankPathScore, InitialPathScore

def Prune(PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore, BeamWidth):
    PrunedBlankPathScore = {}
    PrunedPathScore = {}
    scorelist = []
    # First gather all the relevant scores
    for p in PathsWithTerminalBlank:
        scorelist.append(BlankPathScore[p])

    for p in PathsWithTerminalSymbol:
        scorelist.append(PathScore[p])

    # Sort and find cutoff score that retains exactly BeamWidth paths
    # sort(scorelist) # In decreasing order
    scorelist.sort(reverse = True)    
    # cutoff = BeamWidth < length(scorelist) ? scorelist[BeamWidth] : scorelist[end]

    if BeamWidth >= len(scorelist):
        cutoff = scorelist[-1]
    else:
        cutoff = scorelist[BeamWidth]

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

def ExtendWithBlank(PathsWithTerminalBlank, PathsWithTerminalSymbol, y, BlankPathScore, PathScore):
    UpdatedPathsWithTerminalBlank = set() 
    UpdatedBlankPathScore = {}
    # First work on paths with terminal blanks
    #(This represents transitions along horizontal trellis edges for blanks)
    for path in PathsWithTerminalBlank:
        # Repeating a blank doesn’t change the symbol sequence
        UpdatedPathsWithTerminalBlank.add(path) # Set addition
        UpdatedBlankPathScore[path] = BlankPathScore[path] * y[0]

    # Then extend paths with terminal symbols by blanks
    for path in PathsWithTerminalSymbol:
        # If there is already an equivalent string in UpdatesPathsWithTerminalBlank
        # simply add the score. If not create a new entry
        if path in UpdatedPathsWithTerminalBlank:
            UpdatedBlankPathScore[path] += PathScore[path]* y[0]
        else:
            UpdatedPathsWithTerminalBlank.add(path) # Set addition
            UpdatedBlankPathScore[path] = PathScore[path] * y[0]

    return UpdatedPathsWithTerminalBlank, UpdatedBlankPathScore

def ExtendWithSymbol(PathsWithTerminalBlank, PathsWithTerminalSymbol, SymbolSet, y, BlankPathScore, PathScore):
    UpdatedPathsWithTerminalSymbol = set()  
    UpdatedPathScore = {}
    # First extend the paths terminating in blanks. This will always create a new sequence
    for path in PathsWithTerminalBlank:
        index = 1
        for c in SymbolSet: # SymbolSet does not include blanks
            newpath = path + c # Concatenation
            UpdatedPathsWithTerminalSymbol.add(newpath) # Set addition
            UpdatedPathScore[newpath] = BlankPathScore[path] * y[index]
            index+=1

    # Next work on paths with terminal symbols
    for path in PathsWithTerminalSymbol:
        # Extend the path with every symbol other than blank
        index = 1
        for c in SymbolSet: # SymbolSet does not include blanks
            # newpath = (c == path[end]) ? path : path + c # Horizontal transitions don’t extend the sequence
            if c == path[-1]:
                newpath = path
            elif c != path[-1]:
                newpath = path + c
            # newpath = (c == path[end]) ? path : path + c # Horizontal transitions don’t extend the sequence
            if newpath in UpdatedPathsWithTerminalSymbol: # Already in list, merge paths
                UpdatedPathScore[newpath] += PathScore[path] * y[index] 
            else: # Create new path
                UpdatedPathsWithTerminalSymbol.add(newpath) # Set addition
                UpdatedPathScore[newpath] = PathScore[path] * y[index] 
            index+=1

    return UpdatedPathsWithTerminalSymbol, UpdatedPathScore

def MergeIdenticalPaths(PathsWithTerminalBlank, PathsWithTerminalSymbol, PathScore, BlankPathScore):
    # All paths with terminal symbols will remain
    MergedPaths = PathsWithTerminalSymbol
    FinalPathScore = PathScore
    # Paths with terminal blanks will contribute scores to existing identical paths from
    # PathsWithTerminalSymbol if present, or be included in the final set, otherwise
    for p in PathsWithTerminalBlank:
        if p in MergedPaths:
            FinalPathScore[p] += BlankPathScore[p]
        else:
            MergedPaths.add(p) # Set addition
            FinalPathScore[p] = BlankPathScore[p]

    return MergedPaths, FinalPathScore
