import math
def isPrime(n):
    for i in range(2,int(math.sqrt(n))+1):
        r=n%i
        if r==0:
            return False
    return True


for i in range(2,32768):
    if isPrime(i):
        print(i,'is Prime')
    else:
        print(i,'is not Prime.')


