from matplotlib import pyplot as plt
import numpy as np
import csv

def ReadCSV(filename):
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        x_mylist = []
        y_mylist = []
        for row in csv_reader:
            x_mylist.append( float(row[0]) )
            y_mylist.append( float(row[1]) )
    return x_mylist, y_mylist

# ******************
_x, _y = ReadCSV('ParabolaWithNoiseOutlier.csv')
number_of_samples = len(_x)
x_pos = np.array(_x)
x_pos = x_pos.reshape(number_of_samples, 1)
y_pos = np.array(_y)
y_pos = y_pos.reshape(number_of_samples, 1)

# *****************************
plt.scatter(
   x_pos, y_pos, color="gold", marker=".", label="Training data with noise and outliers"
)

plt.xlabel('x', fontsize=15)
plt.ylabel('y', fontsize=15)

# *************************** 

t1 = x_pos ** 2
t1 = t1.reshape(number_of_samples, 1)
t2 = x_pos.reshape(number_of_samples, 1)
K = np.concatenate( (t1, t2), axis=1 )

# ************************
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(K, y_pos)

print( f'Linear regression model: Coefficients = {reg.coef_}  intercept = {reg.intercept_}' )
print( f' y = {reg.coef_[0][0]} * x**2 + {reg.coef_[0][1]} * x + {reg.intercept_}' )

# ****************************
#Draw the fitted curve
min_x = min(x_pos)
max_x = max(x_pos)

numOfPoint = 300
temp_x = np.linspace(min_x, max_x, num = numOfPoint)
xpoints = temp_x.reshape(numOfPoint, 1)
xxpoints = xpoints ** 2
H = np.concatenate( (xxpoints, xpoints), axis=1)
prediction = reg.predict(H)

plt.plot(xpoints, prediction, color="blue", linewidth=1, label="LR model")


plt.legend(loc='lower left', fontsize=12)
plt.title('Linear Regression - data with noise and outliers')
plt.show()
# ******************************
prediction = reg.predict( K )
residual = prediction - y_pos
plt.hist(residual, bins=30)
plt.xlabel('Residual', fontsize=15)
plt.ylabel('Count', fontsize=15)
plt.title('Residual histogram (data with noise and outliers)')
plt.show()

# ****************************
# Predict y value given x value of 40.0

x_test = np.array([[40.0 ** 2, 40.0],])
y_result = reg.predict(x_test)

print(f'When x = {x_test[0]}, y = {y_result[0]}')