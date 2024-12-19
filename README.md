# Unified Hasher
This library aims to streamline usage of different perceptual image hashes. 

# Installation
**pip**
```pip install unihasher```

# Details
The library provides the following methods for comparing the similarity of two image hashes:
1. Individual Hash - The verdict for whether an image was good or modified from a bad one was determined solely from a single hash algorithm.
2. Majority Decision - The similarity values for all four hashing algorithms were compared separately, and the final verdict was the verdict of the majority of the hash algorithms. In the case of a tie, the verdict of the best performing hash from Approach 1 was taken. 
3. Decision Tree - The similarity values for a combination of all four hashing algorithms were considered by passing the values through a decision tree.

The hashing algorithms implemented are:
dhash, phash, whash from imagehash library
nmfhash adapted from Robust Perceptual Image Hashing Based on Ring Partition and NMF (Tang et al.)

For more details, please refer to our paper.

Made by: [Akshara Mantha](https://github.com/mynameisashllee), [Ruijia Peng](https://github.com/mango-milkshake), [Tan Siying](https://github.com/Cp-Youfy)

# Usage
Please refer to `unihasher_demo/unihasher_usage.py` for details on how you may use the library.