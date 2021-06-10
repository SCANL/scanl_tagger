def powerset(s):
    x = len(s)
    for i in range(1 << x):
        print([s[j] for j in range(x) if (i & (1 << j))])
#['CONTEXT', 'WORD', 'TYPE', 'NORMALIZED_POSITION', 'POSITION', 'MAXPOSITION']
powerset(['TYPE', 'PARAMETERS', 'DECLARATIONS', 'LINES', 'RETURNSIZE', 'PARAMAGGLOMSIZE', 'SURROUNDINGSIZE'])