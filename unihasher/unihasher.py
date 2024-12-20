from .hash import Hasher
import imagehash

# Number of hashes implemented -- change if extending library
NUM_HASHES = 4

class Unihasher:
    '''
    Provides methods to use hashes in this library
    === TO USE ===
    Run set_thresholds(dhashThresh, phashThresh, whashThresh, nmfhashThresh) first to set the hash thresholds as desired, or leave parameters blank to use default configurations from our paper.
    '''
    def __init__(self, imgHashSize:int=16, nmfHashSize:int=512, nmfHashRings:int=32):
        '''
        Initialises parameters for hashes
        imgHashSize: Size to downscale phash, dhash and whash to (n x n)
        nmfHashSize: Size to downscale nmfhash to (n x n)
        nmfHashRings: Number of rings to use for nmfhash
        '''
        self.imgHashSize = imgHashSize
        self.nmfHashSize = nmfHashSize
        self.nmfHashRings = nmfHashRings
        self.hasher = Hasher(imgHashSize)

    def set_thresholds(self, dhashThresh:float=0.334, phashThresh:float=0.348, whashThresh:float=0.191, nmfhashThresh:float=0.952):
        self.hashThreshDict = {
            'dhash': dhashThresh,
            'phash': phashThresh,
            'whash': whashThresh,
            'nmfhash': nmfhashThresh
        }


    def single_hash(self, hashType:str, imgPath:str, toStr:bool=True):
        '''
        Hashes image with one hash
        hashType: dhash | phash | whash | nmfhash
        imgPath: Path to image file
        toStr: (for imagehash dhash | phash | whash) Store as hex string
        '''

        if hashType == 'dhash':
            if toStr:
                return str(self.hasher.dhash(imgPath, self.imgHashSize))
            
            return self.hasher.dhash(imgPath, self.imgHashSize)
        
        if hashType == 'phash':
            if toStr:
                return str(self.hasher.phash(imgPath, self.imgHashSize))
            return self.hasher.phash(imgPath, self.imgHashSize)

        if hashType == 'whash':
            if toStr:
                return str(self.hasher.whash(imgPath, self.imgHashSize))
            return self.hasher.whash(imgPath, self.imgHashSize)

        if hashType == 'nmfhash':
            return self.hasher.nmfhash(imgPath, self.nmfHashSize, self.nmfHashRings)
        
        raise Exception("single_hash: Invalid hash type provided")
    
    def gen_all_hashes(self, imgPath:str, toStr:bool=True):
        '''
        Returns a tuple of hashes (dhash, phash, whash, nmfhash) of the input image
        imgPath: Path to image file
        toStr: (for imagehash dhash | phash | whash) Store as hex string
        '''
        if toStr:
            return (str(self.hasher.dhash(imgPath, self.imgHashSize)), str(self.hasher.dhash(imgPath, self.imgHashSize)), str(self.hasher.whash(imgPath, self.imgHashSize)), self.hasher.nmfhash(imgPath, self.nmfHashSize, self.nmfHashRings))
        return (self.hasher.dhash(imgPath, self.imgHashSize), self.hasher.phash(imgPath, self.imgHashSize), self.hasher.whash(imgPath, self.imgHashSize), self.hasher.nmfhash(imgPath, self.nmfHashSize, self.nmfHashRings))

    def comp_hashes(self, hashType:str, h1:str, h2:str) -> float:
        '''
        Returns similarity metric for the two hashes given hashType
        hashType: dhash | phash | whash | nmfhash
        h1: Hash string 1
        h2: Hash string 2 
        '''

        if hashType == 'nmfhash':
            return self.hasher.pearsonCorr(h1, h2)
        
        # imagehash hamming
        h1 = imagehash.hex_to_hash(h1)
        h2 = imagehash.hex_to_hash(h2)

        return self.hasher.hamming(h1, h2)

    def single_hash_comp(self, hashType:str, h1:str, h2:str) -> bool:
        '''
        Returns True / False if two hash strings are matching based on hash type given
        hashType: dhash | phash | whash | nmfhash
        h1: Hash string 1
        h2: Hash string 2 
        '''
        
        simMetric = self.comp_hashes(hashType, h1, h2)

        # Check for existence of hashThreshDict
        try:
            self.hashThreshDict
        except:
            raise Exception("Please set thresholds using set_thresholds() first")

        try:
            # nmfhash result
            if hashType == 'nmfhash':
                return simMetric > self.hashThreshDict[hashType]
            
            # imagehash result            
            return simMetric < self.hashThreshDict[hashType]
        except:
            raise Exception("single_hash_comp: Invalid hash type provided")
        
    def majority_hash_comp(self, imgPath1:str, imgPath2:str, thresh:int=2, verbose:bool=False) -> bool:
        '''
        Returns True if **more than** thresh hashes match. If tie, use dhash result.
        h1: Hash string 1
        h2: Hash string 2 
        verbose: Prints output of hashes that are matching
        '''
        try:
            assert 0 <= thresh < NUM_HASHES
        except:
            raise Exception("thresh must be within range [0, 4) as there are 4 hashes available.")

        # Check for existence of hashThreshDict
        try:
            self.hashThreshDict
        except:
            raise Exception("Please set thresholds using set_thresholds() first")
        
        isMatching = 0

        # Initialise dictionary to store the similarities
        simDict = self.gen_all_sim(imgPath1, imgPath2)  

        # Checks simDict is populated
        try:
            assert len([num for num in simDict.values() if num == -2]) == 0
        except:
            raise Exception("decision_tree_comp: Error in populating simDict")

        dhashResult = False
        
        for hashType, sim in simDict.items():
            if hashType in ['dhash', 'phash', 'whash']:
                result = self.hashThreshDict[hashType] >= sim
                isMatching += int(result)
            elif hashType == 'nmfhash':
                result = self.hashThreshDict[hashType] <= sim
                isMatching += int(result)
            else:
                raise Exception("decision_tree_comp: Error in hashThreshDict keys")

            if verbose:
                print(f"{hashType} was matching? {result}")

        if isMatching == thresh:
            return dhashResult
        else:
            return isMatching > thresh
    
    def gen_all_sim(self, imgPath1: str, imgPath2: str):

        dhash1 = self.hasher.dhash(imgPath1)
        dhash2 = self.hasher.dhash(imgPath2)
        dham = self.hasher.hamming(dhash1, dhash2)

        phash1 = self.hasher.phash(imgPath1)
        phash2 = self.hasher.phash(imgPath2)
        pham = self.hasher.hamming(phash1, phash2)    

        whash1 = self.hasher.whash(imgPath1)
        whash2 = self.hasher.whash(imgPath2)
        wham = self.hasher.hamming(whash1, whash2)    

        nmfhash1 = self.hasher.nmfhash(imgPath1)
        nmfhash2 = self.hasher.nmfhash(imgPath2)
        nmfcorr = self.hasher.pearsonCorr(nmfhash1, nmfhash2)

        return {
            'dhash': dham,
            'phash': pham,
            'whash': wham,
            'nmfhash': nmfcorr
        }

    def gen_all_sim_data(self, hashResults1: tuple, hashResults2: tuple):
        '''
        Generates the similarity scores given two hash tuples representing two images
        hashResults1: tuple of precomputed hashes of the first image in format (dhash, phash, whash, nmfhash)
        hashResults2: tuple of precomputed hashes of the second image in format (dhash, phash, whash, nmfhash)
        Returns dictionary of hash similarities that can be passed into test_decision_tree_comp
        '''

        dham = self.hasher.hamming(hashResults1[0], hashResults2[0])
        pham = self.hasher.hamming(hashResults1[1], hashResults2[1])
        wham = self.hasher.hamming(hashResults1[2], hashResults2[2])
        nmfcorr = self.hasher.hamming(hashResults1[3], hashResults2[3])

        return {
            'dhash': dham,
            'phash': pham,
            'whash': wham,
            'nmfhash': nmfcorr
        }

    def decision_tree_comp(self, imgPath1:str, imgPath2:str, verbose:bool=False) -> bool:
        # modify this to take the similarity as input -- write separate command to generate all hash variant similarities and optionally write to a csv
        '''
        Returns True / False based on the result of the decision tree
        (True: Matching image)\n
        Please modify tree accordingly to any needs or new discoveries
        imgPath1: Path of first image to match
        imgPath2: Path of second image to match
        verbose: Prints each node dictionary to trace traversal path

        NOTE: Thresholds used in tree are fixed, not using self.hashThreshDict! They are optimised through our decision tree construction. Please refer to our Report for the visualisation.
        '''

        # Initialise dictionary to store the similarities
        simDict = self.gen_all_sim(imgPath1, imgPath2)  

        # Checks simDict is populated
        try:
            assert len([num for num in simDict.values() if num == -2]) == 0
        except:
            raise Exception("decision_tree_comp: Error in populating simDict")
        
        return self.test_decision_tree_comp(simDict, verbose)
    
    def test_decision_tree_comp(self, simDict:dict, verbose=False):
        '''
        Returns True / False based on the result of the decision tree
        (True: Matching image)\n
        Use this function if you have similarity metrics to check against the decision tree\n
        Also acts as a helper function for decision_tree_comp
        simVals: Dictionary of similarity score e.g. {'dhash': 0, 'phash': 0, 'whash': 0, 'nmfhash': 0}
        verbose: Prints each node dictionary to trace traversal path
        '''
        # Decision tree code (tree[0] is root, id indicates index)
        tree = (
            { 'id': 0, 'condition': simDict['dhash'] <= 0.334, 'trueNode': 1, 'falseNode': 3},
            { 'id': 1, 'condition': simDict['whash'] <= 0.277, 'trueNode': True, 'falseNode': 2},
            { 'id': 2, 'condition': simDict['nmfhash'] <= 0.829, 'trueNode': False, 'falseNode': True},
            { 'id': 3, 'condition': simDict['nmfhash'] <= 0.976, 'trueNode': 4, 'falseNode': True},
            { 'id': 4, 'condition': simDict['phash'] <= 0.347, 'trueNode': True, 'falseNode': False}
        )
        
        currentNode = tree[0]
        # Condition to recursively traverse tree while not encountering a result
        while True:
            if verbose: print(currentNode)
            result = currentNode['condition']
            if result:
                # True
                if type(currentNode['trueNode']) == int:
                    currentNode = tree[currentNode['trueNode']]
                else:
                    return currentNode['trueNode']
            else:
                # False
                if type(currentNode['falseNode']) == int:
                    currentNode = tree[currentNode['falseNode']]
                else:
                    return currentNode['falseNode']


    def evaluate(self, tp, tn, fp, fn):
        '''
        Function to check the results if using this library for testing
        '''
        print(f"True Positive: {tp}")
        print(f"False Positive: {fp}")
        print(f"True Negative: {tn}")
        print(f"False Negative: {fn}")

        # Calculate metrics for evaluation
        # Accuracy: overall effectiveness considering both pos/neg
        accuracy = (tp+tn)/(tp+tn+fp+fn)

        # Precision: how many predicted pos/neg are actually pos/neg
        precision_pos = tp/(tp+fp)
        precision_neg = tn/(tn+fn)

        # Recall: how many of the actual pos/neg were identified by model
        recall_pos = tp/(tp+fn)
        recall_neg = tn/(tn+fp)

        # F1: balancing precision and recall
        f1_pos = 2*precision_pos*recall_pos/(precision_pos+recall_pos)
        f1_neg = 2*precision_neg*recall_neg/(precision_neg+recall_neg)

        print(f"Accuracy: {accuracy}")
        print(f"Precision (Positive): {precision_pos}")
        print(f"Precision (Negative): {precision_neg}")
        print(f"Recall (Positive): {recall_pos}")
        print(f"Recall (Negative): {recall_neg}")
        print(f"F1 Score (Positive): {f1_pos}")
        print(f"F1 Score (Negative): {f1_neg}")
