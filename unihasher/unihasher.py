from hash import Hasher
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
                return str(Hasher.dhash(imgPath, self.imgHashSize))
            
            return Hasher.dhash(imgPath, self.imgHashSize)
        
        if hashType == 'phash':
            if toStr:
                return str(Hasher.dhash(imgPath, self.imgHashSize))
            return Hasher.phash(imgPath, self.imgHashSize)

        if hashType == 'whash':
            if toStr:
                return str(Hasher.dhash(imgPath, self.imgHashSize))
            return Hasher.whash(imgPath, self.imgHashSize)

        if hashType == 'nmfhash':
            return Hasher.nmfhash(imgPath, self.nmfHashSize, self.nmfHashRings)
        
        raise Exception("single_hash: Invalid hash type provided")
    
    def comp_hashes(self, hashType:str, h1:str, h2:str) -> float:
        '''
        Returns similarity metric for the two hashes given hashType
        hashType: dhash | phash | whash | nmfhash
        h1: Hash string 1
        h2: Hash string 2 
        '''

        if hashType == 'nmfhash':
            return Hasher.pearsonCorr(h1, h2)
        
        # imagehash hamming
        h1 = imagehash.hex_to_hash(h1)
        h2 = imagehash.hex_to_hash(h2)

        return Hasher.hamming(h1, h2)

    def single_hash_comp(self, hashType:str, h1:str, h2:str) -> bool:
        '''
        Returns True / False if two hash strings are matching based on hash type given
        hashType: dhash | phash | whash | nmfhash
        h1: Hash string 1
        h2: Hash string 2 
        '''
        
        simMetric = self.comp_hashes(hashType, h1, h2)

        try:
            # nmfhash result
            if hashType == 'nmfhash':
                return simMetric > self.hashThreshDict[hashType]
            
            # imagehash result            
            return simMetric < self.hashThreshDict[hashType]
        except:
            raise Exception("single_hash_comp: Invalid hash type provided")
        
    def majority_hash_comp(self, h1:str, h2:str, thresh:int=2) -> bool:
        '''
        Returns True if **more than** thresh hashes match. If tie, use dhash result.
        h1: Hash string 1
        h2: Hash string 2 
        '''
        try:
            assert 0 <= thresh < NUM_HASHES
        except:
            raise Exception("thresh must be within range [0, 4) as there are 4 hashes available.")

        isMatching = 0
        dhashResult = False
        
        for hashType in ['dhash', 'phash', 'whash', 'nmfhash']:
            isMatching += int(self.single_hash_comp(hashType, h1, h2))
            if hashType == 'dhash':
                dhashResult = self.single_hash_comp(hashType, h1, h2)

        if isMatching == thresh:
            return dhashResult
        else:
            return isMatching > thresh
    
    def gen_all_sim(self, imgPath1: str, imgPath2: str):

        dhash1 = Hasher.dhash(imgPath1)
        dhash2 = Hasher.dhash(imgPath2)
        dham = Hasher.hamming(dhash1, dhash2)

        phash1 = Hasher.phash(imgPath1)
        phash2 = Hasher.phash(imgPath2)
        pham = Hasher.hamming(phash1, phash2)    

        whash1 = Hasher.whash(imgPath1)
        whash2 = Hasher.whash(imgPath2)
        wham = Hasher.hamming(whash1, whash2)    

        nmfhash1 = Hasher.nmfhash(imgPath1)
        nmfhash2 = Hasher.nmfhash(imgPath2)
        nmfcorr = Hasher.pearsonCorr(nmfhash1, nmfhash2)

        return {
            'dhash': dham,
            'phash': pham,
            'whash': wham,
            'nmfhash': nmfcorr
        }


    def decision_tree_comp(self, imgPath1:str, imgPath2:str) -> bool:
        # modify this to take the similarity as input -- write separate command to generate all hash variant similarities and optionally write to a csv
        '''
        Returns True / False based on the result of the decision tree
        (True: Matching image)
        Please modify tree accordingly to any needs or new discoveries
        imgPath1: Path of first image to match
        imgPath2: Path of second image to match

        NOTE: Thresholds used in tree are fixed, not using self.hashThreshDict! They are optimised through our decision tree construction. Please refer to our Report for the visualisation.
        '''

        # Initialise dictionary to store the similarities
        simDict = self.gen_all_sim(imgPath1, imgPath2)  

        # Checks simDict is populated
        try:
            assert len([num for num in simDict.values() if num == -2]) == 0
        except:
            raise Exception("decision_tree_comp: Error in populating simDict")
        
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
        while type(currentNode) != bool:
            result = currentNode['condition']
            if result:
                # True
                currentNode = tree[currentNode['trueNode']]
            else:
                # False
                currentNode = tree[currentNode['falseNode']]
        
        # Result (True/False for matching)
        return currentNode
    
    def evaluate(tp, tn, fp, fn):
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
