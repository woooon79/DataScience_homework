import numpy as np
import matplotlib.pyplot as plt
import math
import random

# np.random 사용
wt = np.random.uniform(40.0, 90.0, size=100)  # wt.dtype=float64
ht = np.random.randint(140, 200, size=100)  # 센치미터단위 #ht.dtype=int32
ht2 = ht / 100  # 미터단위 ht2.dtype=float64

print('wt:', wt[:10])
print('ht:', ht[:10])

BMI = wt / (ht2 * ht2)
print('BMI:', BMI)


def makePlotData(arr):
    Underweight = list()
    Healthy = list()
    Overweight = list()
    Obese = list()
    for i in range(0, 100):
        if BMI[i] < 18.5:
            Underweight.append(arr[i])
        elif 18.5 <= BMI[i] and BMI[i] <= 24.9:
            Healthy.append(arr[i])
        elif 25 <= BMI[i] and BMI[i] <= 29.9:
            Overweight.append(arr[i])
        else:
            Obese.append(arr[i])
    plotData = [np.array(Underweight), np.array(Healthy), np.array(Overweight), np.array(Obese)]
    return plotData


plotData = makePlotData(wt)
plt.boxplot(plotData, labels=['Underweight', 'Healthy', 'Overweight', 'Obese'])
plt.ylabel('weight')
plt.title('Boxplot-weight')
plt.show()

plotData = makePlotData(ht)
plt.boxplot(plotData, labels=['Underweight', 'Healthy', 'Overweight', 'Obese'])
plt.ylabel('height')
plt.title('Boxplot-height')
plt.show()


