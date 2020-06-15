import csv
from os import path
from symspellpy import SymSpell, Verbosity
import pickle
import re
import json


# Custom version of SymsSpell class is created to allow a few extra characters to be accepted in corpus entries
class SymSpellMTG(SymSpell):
    def _parse_words(self, text):
        # \W = Alphanumeric characters, including non-latin characters, umlaut characters and digits.
        # Also allow charactes '’,-_
        matches = re.findall(r"(([^\W]|['’,-_])+)", text)
        matches = [match[0] for match in matches]
        return matches


# Ready the card database into a corpus format used by Symspell
def createCorpus(corpusPath, cardNames):
    corpusStr = " ".join(cardNames)
    f = open(corpusPath, "w", encoding="utf8")
    f.write(corpusStr)
    f.close()


# Read unique card names into a list
def readNames(dataPath):
    with open(dataPath) as json_file: 
        cards = json.load(json_file) 
        names = []

        # Drop the last part of dual names - anything after //
        for card in cards:
            name = card["name"]
            simpleName = name.split("//")[0].rstrip()
            simpleName = simpleName.replace("Æther", "Aether")
            names.append(simpleName)
    return list(set(names)) 


''' Create a set of all partial names on word level, starting from the first word.
Example:
input = ["Black Lotus", "Black Mana Battery", "Black Oak of Odunos"]
output = ["Black", "Black Lotus", "Black Mana", "Black Mana Battery", "Black Oak", "Black Oak of", "Black Oak of Odunos"]
'''
def buildPartialNames(names):
    sep = " "
    collector = []
    for name in names:
        parts = name.split(sep)
        for i in range(len(parts)):
            selParts = parts[0:(i+1)]
            joinedSelParts = sep.join(selParts)
            collector.append(joinedSelParts)
    return set(collector)     


class SymspellMTGNames:
    CARD_DATA_PATH = "symspell_data/cards.json"
    CORPUS_PATH = "symspell_data/corpus.txt"
    NAME_SET_PATH = "symspell_data/name_set.pickle"
    cardNames = readNames(CARD_DATA_PATH)
    cardNames = set(cardNames)


    def __init__(self):
        if not path.isfile(self.CORPUS_PATH):
            print("Create corpus for Symspell")
            createCorpus(self.CORPUS_PATH, self.cardNames)
        else:
            print("Using existing corpus in '", self.CORPUS_PATH, "' for Symspell")    

        if not path.isfile(self.NAME_SET_PATH):
            print("Building name part set for Symspell")
            self.partialNames = buildPartialNames(self.cardNames)
            with open(self.NAME_SET_PATH, "wb") as wFile:
                pickle.dump(self.partialNames, wFile, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            print("Existing name part set found in '", self.NAME_SET_PATH, "', using it.")
            with open(self.NAME_SET_PATH, "rb") as rFile:
                self.partialNames = pickle.load(rFile)

        self.instance = SymSpellMTG(max_dictionary_edit_distance=3, prefix_length=7, count_threshold=1)
        self.instance.create_dictionary(self.CORPUS_PATH, encoding="utf-8")


    # Do the name correction for a raw recognition result
    def lookup(self, name):
        parts = name.split(" ")

        # Create a separate suggestion list for each word
        suggestions = []
        for part in parts:
            wSuggestions = self.instance.lookup(part, max_edit_distance=2, verbosity=Verbosity.ALL)
            suggestions.append(wSuggestions)

        candidates = []
        for level in range(len(parts)):
            levelSuggestions = suggestions[level]
            levelCandidates = []

            # Go through candidates for each word so that new candidates are always starts of names or names
            for suggestion in levelSuggestions:
                if level == 0:
                    if suggestion.term in self.partialNames:
                        candDict = {"term": suggestion._term, "distance": suggestion._distance}
                        levelCandidates.append(candDict)
                else:
                    for candidate in candidates:
                        prefix = candidate["term"]
                        suffix = suggestion.term
                        newTerm = prefix + " " + suffix
                        if newTerm in self.partialNames:
                            candDict = {"term": newTerm, "distance": candidate["distance"] + suggestion.distance}
                            levelCandidates.append(candDict)

            candidates = levelCandidates


        # Finally select the candidate with least total edit distance
        while (len(candidates) > 0):
            minCandidate = min(candidates, key=lambda cand: cand["distance"])
            candidates.remove(minCandidate)
            dist = minCandidate["distance"]
            candTerm = minCandidate["term"]
            if candTerm in self.cardNames:
                return candTerm, dist
        return None, 0        


if __name__ == "__main__":
    test = SymspellMTGNames()
    print(test.lookup("Black Zat"))
    print(test.lookup("Nyx-Fleece Ram"))
    print(test.lookup("Hallar, the Firefletcher"))
    print(test.lookup("ade into antiquity"))
    print(test.lookup("honor the god-pharaoh"))