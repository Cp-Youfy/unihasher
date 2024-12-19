import imagehash
from .NMF_Hash import NMFHash
from PIL import Image

class Hasher:
    '''
    Hasher class encapsulates all available hashes
    dhash, phash, whash, nmfhash
    '''

    def __init__(self, imgHashSize:int=16):
        '''
        imgHashSize: Size used for normalisation of imagehash'''
        self.imgHashSize = imgHashSize
        self.MAX_IMGHASH_SIZE = imgHashSize ** 2

    def dhash(self, imgPath:str, hashSize:int=16):
        '''
        Hashes the image using dhash from imagehash library
        imgPath: Path to image
        hashSize: Size to downsize image to ((n + 1) x n square)
        '''
        img = Image.open(imgPath)
        return imagehash.dhash(img, hashSize)

    def phash(self, imgPath:str, hashSize:int=16):
        '''
        Hashes the image using phash from imagehash library
        imgPath: Path to image
        hashSize: Size to downsize image to (n x n square)
        '''
        img = Image.open(imgPath)
        return imagehash.phash(img, hashSize)

    def whash(self, imgPath:str, hashSize:int=16):
        '''
        Hashes the image using whash from imagehash library
        imgPath: Path to image
        hashSize: Size to downsize image to (n x n square)
        '''
        img = Image.open(imgPath)
        return imagehash.whash(img, hashSize)

    def nmfhash(self, imgPath:str, hashSize:int=512, ringNo:int=32):
        '''
        Hashes the image using NMF Hash Python implementation
        '''
        try:
            # Checks for stored instance of NMF hash with the same parameters
            self.nmfHashObj
            assert self.nmfHashObj.imgSize == hashSize
            assert self.nmfHashObj.ring == ringNo
        except:
            # Creates / updates the NMF hash object to avoid creating unnecessary duplicate instances of the hash object
            self.nmfHashObj = NMFHash(hashSize, ringNo)

        return self.nmfHashObj.singleHashCode(imgPath)

    def hamming(self, h1:imagehash.ImageHash, h2:imagehash.ImageHash):
        '''
        Calculates hamming distance between two imagehash hashes (dhash, phash, whash)
        h1: ImageHash object (use imagehash.hex_to_hash(hexStr) to convert from stored hex string)
        h2: ImageHash object (use imagehash.hex_to_hash(hexStr) to convert from stored hex string)
        Returns value in range [0, 1]: the closer to 0, the greater similarity
        '''
        return (h2 - h1) / self.MAX_IMGHASH_SIZE
    
    def pearsonCorr(self, h1:str, h2:str):
        '''
        Calculates Pearson correlation coefficient for two NMF hashes
        h1: Hash string 1
        h2: Hash string 2
        Returns value in range [-1, 1]: the closer to 1, the greater similarity
        '''
        return self.nmfHashObj.compareHashString(h1, h2)