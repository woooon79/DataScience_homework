def makeDict(K,V):
    s=len(K)
    D=dict()
    for i in range(0,s):
        D[K[i]]=V[i]

    return D



K = ('Korean', 'Mathematics', 'English')
V = (90.3, 85.5, 92.7)
D =makeDict(K,V)

for key,val in D.items():
    print("key = {key}, value = {value}".format(key=key,value=val))