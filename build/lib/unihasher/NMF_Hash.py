import numpy as np
import cv2

class NMFHash:
    '''
    Adapted from "Robust Perceptual Image Hashing Based on Ring Partition and NMF"
    (https://ieeexplore.ieee.org/document/6477042)
    '''
    def __init__(self, imgSize:int=512, ring:int=32, rank:int=2):
        '''
        imgSize: Size of image to downscale to
        ring: Number of rings to add to image
        '''
        self.imgSize = imgSize
        self.ring = ring
        self.rank = rank
    
    def time_convert(self, sec):
        mins = sec // 60
        sec = sec % 60
        hours = mins // 60
        mins = mins % 60
        return "Time Lapsed = {0}:{1}:{2}".format(int(hours),int(mins),sec)

    def nmfFunc(self, v, max_iter):
        '''
        NMF Matrix Factorisation using Multiplicative Update Method
        '''
        n, m = v.shape
        random_state = np.random.RandomState(seed=26)
        w = random_state.uniform(0, 1, (n, self.rank)).astype(np.float32)
        h = random_state.uniform(0, 1, (self.rank, m)).astype(np.float32)

        for _ in range(max_iter):
            # Stage 1
            e1 = np.dot(w, h)
            s1 = np.sum(w, axis=0).reshape(self.rank, 1) # # sum(w) over the rows (rank dimension)
            e2 = v / (e1 + 1e-4) # to avoid division by zero
            s2 = s1 * np.ones((1, m))
            e3 = np.dot(w.T, e2)
            e4 = h * e3
            h = e4 / (s2 + 1e-4)

            # Stage 2
            e1 = np.dot(w, h)
            s1 = np.sum(h, axis=1).reshape(1, self.rank) # sum(h) over the columns (rank dimension)
            e2 = v / (e1 + 1e-4)
            s2 = np.ones((n, 1)) * s1
            e3 = np.dot(e2, h.T)
            e4 = w * e3
            w = e4 / (s2 + 1e-4)
        
        return h

    def ringNMF(self, img):
        '''
        Performing ring NMF on image
        '''
        # Converting image 
        dim = min(img.shape[0], img.shape[1], 512)  # resize
        imgResized = cv2.resize(img, (dim, dim))
        self.imgSize = dim
        # imgResized = cv2.resize(img, (self.imgSize, self.imgSize))
        imgBlurred = cv2.GaussianBlur(imgResized, (3, 3), 1)
        imgYCbCr = cv2.cvtColor(imgBlurred, cv2.COLOR_BGR2YCrCb)

        # Showing converted image
        # cv2.imshow('image window', imgYCbCr)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        i1 = imgYCbCr[:, :, 0]
        cc = self.imgSize / 2.0 + 0.5 if self.imgSize % 2 == 0 else (self.imgSize + 1) / 2.0
        
        rpt = np.arange(1, self.imgSize + 1).astype(np.float32)
        xa = np.repeat(rpt[np.newaxis, :], self.imgSize, axis=0)
        ya = xa.T

        xa = np.power((xa - cc), 2) + np.power((ya - cc), 2)

        rn = np.zeros((1, self.ring), dtype=np.float32)
        for i in range(self.ring):
            rn[0, i] = i + 1
        rn[0, self.ring - 1] = self.imgSize - cc

        s = np.floor(np.pi * rn[0, self.ring - 1] ** 2 / self.ring)
        rn[0, 0] = np.sqrt(s / np.pi)

        for i in range(1, self.ring - 1):
            rn[0, i] = np.sqrt((s + np.pi * rn[0, i - 1] ** 2) / np.pi)
        
        rn = rn ** 2

        v = []

        for j in range(self.imgSize):
            for i in range(self.imgSize):
                if xa[j, i] <= rn[0, 0]:
                    v.append(i1[j, i])
        
        vectorHolder = [v]
        _len = len(v)
        row = 0

        matrixV = np.zeros((int(s), self.ring), dtype=np.float32)

        for r in range(self.ring - 1):
            v1 = []
            for j in range(self.imgSize):
                for i in range(self.imgSize):
                    if rn[0, r] < xa[j, i] <= rn[0, r + 1]:
                        v1.append(i1[j, i])
            _len = min(_len, len(v1))
            vectorHolder.append(v1)
        
        for r in range(self.ring):
            tmp = sorted(vectorHolder[r])[:_len]
            row = 0
            for i in range(_len):
                matrixV[row, r] = tmp[i]
                row += 1
        
        max_iter = 60
        h = self.nmfFunc(matrixV, max_iter)
        return h

    def createHashString(self, h1):
        '''
        Generate hash string based on matrix returned from NMF function
        '''
        f1 = np.round(h1).astype(np.uint8).flatten()
        out = ''
        for t in f1:
            for _ in range(3):
                cur = min(93, t)
                out += chr(cur + ord('!'))
                t -= cur
        return out
    
    def singleHashCode(self, imgPath):
        '''
        Performs NMF and creates hash string for one image
        '''
        img = cv2.imread(imgPath)
        if img is None:
            return None
        h1 = self.ringNMF(img)
        h = self.createHashString(h1)
        return h
    
    def compareHashString(self, hash1, hash2):
        '''
        Compares two hash strings
        returns Pearson Correlation Coefficient (value between -1 to 1: the closer to 1, the greater similarity)
        '''
        temp1 = np.array([sum(ord(c) - ord('!') for c in hash1[i*3:(i+1)*3]) for i in range(64)])
        temp2 = np.array([sum(ord(c) - ord('!') for c in hash2[i*3:(i+1)*3]) for i in range(64)])
        sim = np.corrcoef(temp1, temp2)[0, 1]
        return sim