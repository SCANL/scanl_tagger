import csv
from collections import defaultdict

# Generate a more general pattern by removing the last term
def generalize(pattern):
    terms = pattern.split()    
    if len(terms) > 1:
        return ' '.join(terms[:-1])
    else: 
        return pattern

# Find the common prefix between two patterns  
def common_prefix(a, b):
    terms_a = a.split()
    terms_b = b.split()
    
    prefix = []
    for w1, w2 in zip(terms_a, terms_b):
        if w1 == w2:
            prefix.append(w1)
        else:
            break
    
    return ' '.join(prefix)

patterns = defaultdict(list)  # Map prefixes to list of patterns
finalpatterns = defaultdict(list)  # Map prefixes to list of patterns

with open('test.csv') as f:
    reader = csv.DictReader(f)
    
    # Fill the pattern dictionary with a unique list of all patterns and their counts
    for row in reader:
        if row['GRAMMAR PATTERN'] in patterns:
            patterns[row['GRAMMAR PATTERN']] += 1
        else:
            patterns[row['GRAMMAR PATTERN']] = 1
    
    # Iterate through the unique list of patterns
    for pattern, freq in patterns.items():
        
        foundPattern = False #there is no prefix between these
        while_iters = 0
        
        # If this pattern has appeared twice or more, add it to the final patterns dictionary.
        # If it's already in the final patterns dictionary, then be sure to add the frequency 
        # to what's already in the final dictionary
        if freq >= 2:
            if pattern in finalpatterns:
                finalpatterns[pattern]+=freq
            else:
                finalpatterns[pattern]=freq
            continue

        # The pattern occurs 1 at most. Iterate until we fully generalize the pattern by 
        # removing terms until it matches at least one other pattern in the patterns dictionary
        current_pattern = pattern
        while (foundPattern == False):
            # Generalize by removing the rightmost term as long as there is more than 1 term
            if len(current_pattern.split()) > 1:
                current_pattern = generalize(current_pattern)
            # Iterate through the unique list of patterns again, this time looking for patterns 
            # it now matches because it's been generalized
            for patterninner, freqinner in patterns.items():
                # Calculate the prefix of the generalized pattern versus the currenrt pattern in the unique patterns dict
                prefix = common_prefix(patterninner, current_pattern)
                # If we couldn't find a common pattern, prefix is a blank string. Skip.
                if not any([term.isalnum() for term in prefix.split()]):
                    continue
                # We found a match. Add it to finalpatterns.
                elif (len(prefix.split()) >= 1):
                    foundPattern = True
                    if prefix in finalpatterns:
                        finalpatterns[prefix]+=1
                    else:
                        finalpatterns[prefix]=1
                    break
            
# Output pattern prefix groups
finalfreq = 0
for prefix, freq in finalpatterns.items():
    finalfreq+=freq
    print(f'{prefix},{freq}')

print(finalfreq)