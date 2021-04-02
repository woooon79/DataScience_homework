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
print('BMI:', BMI[:10])

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


labels = ['Underweight', 'Healthy', 'Overweight', 'Obese']

# box plot
plotData = makePlotData(ht)
plt.boxplot(plotData, labels=labels)
plt.ylabel('height')
plt.title('Boxplot-height')
plt.show()
plotData = makePlotData(wt)
plt.boxplot(plotData, labels=labels)
plt.ylabel('weight')
plt.title('Boxplot-weight')
plt.show()

# histogram
plotData = makePlotData(ht)
plt.hist(plotData, bins=4)
plt.xlabel('height')
plt.ylabel("frequency")
plt.title('histogram-height')
plt.legend(labels)
plt.show()

plotData = makePlotData(wt)
plt.hist(plotData, bins=4)
plt.xlabel('weight')
plt.ylabel("frequency")
plt.title('histogram-weight')
plt.legend(labels)
plt.show()

# pie chart
plotData = makePlotData(wt)
plt.pie((len(plotData[0]), len(plotData[1]), len(plotData[2]), len(plotData[3])), labels=labels, autopct='%1.2f%%')
plt.title('pie chart')
plt.show()

# scatter plot
plotData_wt = makePlotData(wt)
plotData_ht = makePlotData(ht)
plt.scatter(plotData_wt[0], plotData_ht[0], c='blue')
plt.scatter(plotData_wt[1], plotData_ht[1], c='green')
plt.scatter(plotData_wt[2], plotData_ht[2], c='red')
plt.scatter(plotData_wt[3], plotData_ht[3], c='black')
plt.title('Scatter plot')
plt.legend(labels)
plt.show()













