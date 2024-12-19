from unihasher.unihasher import Unihasher
from unihasher.hash import Hasher

# Usage of Hasher object
'''
Hasher object is a class wrapper for usage of included hashes
'''

hasher = Hasher()
orig_dhash = hasher.dhash(r'orig.jpg')

print(f"Difference hash of orig.jpg: {str(orig_dhash)}")

# Usage of Unihasher object
unihash = Unihasher()

# Using default thresholds
unihash.set_thresholds()

# Single hash result
other_dhash = unihash.single_hash('dhash', r'other.jpg')
print(f"Difference hash of other.jpg: {str(other_dhash)}")

# Hasher().dhash() returns an ImageHash object; we need to convert it to a string for the hash comparison
orig_dhash = str(orig_dhash)

unmatching_diff = unihash.single_hash_comp('dhash', orig_dhash, other_dhash)
print(f"Individual hash (dhash) | orig.jpg VS other.jpg: {unmatching_diff}")

# We want to check the exact hamming distance of the two images
unmatching_hamming = unihash.comp_hashes('dhash', orig_dhash, other_dhash)
print(f"Hamming distance | orig.jpg VS other.jpg: {unmatching_hamming}")

# How about using a majority approach?
is_matching_majority = unihash.majority_hash_comp(r'orig.jpg', r'other.jpg')
print(f"Majority approach | orig.jpg VS other.jpg: {is_matching_majority}")

is_matching_noiseColour = unihash.majority_hash_comp(r'orig.jpg', r'noiseColour.jpg')
print(f"Majority approach | orig.jpg VS noiseColour.jpg: {is_matching_noiseColour}")

# Notice a limitation of the majority approach (NMFHash's ability to detect reflections is limited)
is_matching_reflectedLR = unihash.majority_hash_comp(r'orig.jpg', r'reflectedLR.jpg', verbose=True)
print(f"Majority approach | orig.jpg VS reflectedLR.jpg: {is_matching_reflectedLR}")

# A more comprehensive approach will be the decision tree
is_matching_dtree = unihash.decision_tree_comp(r'orig.jpg', r'other.jpg')
print(f"Decision tree approach | orig.jpg VS other.jpg: {is_matching_dtree}")

is_matching_dtree_reflectedLR = unihash.decision_tree_comp(r'orig.jpg', r'reflectedLR.jpg')
print(f"Decision tree approach | orig.jpg VS reflectedLR.jpg: {is_matching_dtree_reflectedLR}")

# Let's test the decision tree approach on a dataset