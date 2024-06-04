from matplotlib import pyplot as plt
import numpy as np

import csv

# https://scikit-learn.org/stable/auto_examples/linear_model/plot_ransac.html#sphx-glr-auto-examples-linear-model-plot-ransac-py

def ReadCSV(filename):
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        x_mylist = []
        y_mylist = []
        for row in csv_reader:
            x_mylist.append( float(row[0]) )
            y_mylist.append( float(row[1]) )
    return x_mylist, y_mylist

# **********************
plt.close('all')

_x, _y = ReadCSV('Line.csv')
n_samples = len(_x)
x = np.array(_x)
x = x.reshape(n_samples, 1)
y = np.array(_y)
y = y.reshape(n_samples, 1)

#add some noise to y
y = y + np.random.normal(0, 1.5, size=(n_samples, 1))

plt.figure("Figure 1")
plt.scatter(
   x, y, color="orange", marker=".", label="data with noise"
)

#plt.xlabel('Time (s)', fontsize=15)
#plt.ylabel('Speed (m/s)', fontsize=15)
#plt.title('RANSAC regression - data with noise and outliers')
# plt.legend(loc='upper left', fontsize=12)
# plt.show()

# **********************
# Create outlier data
n_outliers = 200
np.random.seed(0)
x_outlier =  np.random.normal(8.0, 1.0, size=(n_outliers, 1))
y_outlier =  np.random.normal(10.0, 0.5, size=(n_outliers, 1))


plt.figure("Figure 1")
plt.scatter(
   x_outlier, y_outlier, color="green", marker=".", label="outliers"
)


#plt.legend(loc='upper left', fontsize=12)
#plt.show()

# *********************************************************
# add outliers to x and y
x = np.concatenate( (x, x_outlier), axis=0)
y = np.concatenate( (y.reshape(n_samples, 1), y_outlier), axis=0 )


# **********************
from sklearn.linear_model import RANSACRegressor

ransac = RANSACRegressor(random_state=0).fit(x, y)
print( f'RANSAC regression model (with noise and outliers): Coefficients = {ransac.estimator_.coef_}  intercept = {ransac.estimator_.intercept_}' )
print( f'y = {ransac.estimator_.coef_[0]} * x  +  {ransac.estimator_.intercept_}' )

inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
# *************************
# Draw the fitted line using RANSAC
min_x = min(x)
max_x = max(x)

numOfPoint = 300
temp_x = np.linspace(min_x, max_x, num = numOfPoint)
xpoints = temp_x.reshape(numOfPoint, 1)
ypoints = ransac.predict(xpoints)


plt.figure("Figure 1")
plt.plot(xpoints, ypoints, color="blue", linewidth=1, label="RANSAC regression model")

plt.xlabel('Time (s)', fontsize=15)
plt.ylabel('Speed (m/s)', fontsize=15)
plt.title('RANSAC regression - data with noise and outliers')
plt.legend(loc='upper left', fontsize=12)
plt.show()
# ******************************
# Extract all the inlier points using the inlier_mask produced by RANSACRegressor
our_inlier_x = x[ inlier_mask ]
our_inlier_y = y[ inlier_mask ]

# *************************** 
# Show all the inliers
plt.figure("Figure 2")
plt.scatter(
   our_inlier_x, our_inlier_y, color="green", marker=".", label="our inliers"
)

plt.xlabel('Time (s)', fontsize=15)
plt.ylabel('Speed (m/s)', fontsize=15)
plt.title('Our inliers')
plt.show()
# ******************************
# Use linear regressor to fit a line with our inliers
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(our_inlier_x, our_inlier_y)

print( f'Linear regression model (using only inliers): Coefficients = {reg.coef_}  intercept = {reg.intercept_}' )
print( f'y = {reg.coef_[0]} * x  +  {reg.intercept_}' )
# ******************************
# Draw the fitted line using linear regressor
min_x = min(our_inlier_x)
max_x = max(our_inlier_x)

numOfPoint = 300
temp_x = np.linspace(min_x, max_x, num = numOfPoint)
xpoints = temp_x.reshape(numOfPoint, 1)
ypoints = reg.predict(xpoints)

plt.figure("Figure 3")
plt.plot(xpoints, ypoints, color="blue", linewidth=1, label="LR model using inliers")

#Show all the data as well
plt.figure("Figure 3")
plt.scatter(
   x, y, color="orange", marker=".", label="data with noise and outliers"
)

plt.legend(loc='upper left', fontsize=12)
plt.xlabel('Time (s)', fontsize=15)
plt.ylabel('Speed (m/s)', fontsize=15)
plt.title('Estimated model using inliers')

plt.show()
# ******************************
# Compute the residual and show it.
prediction = reg.predict( x )
residual = prediction - y
plt.figure("Figure 4")
plt.hist(residual, bins=30)
plt.xlabel('Residual', fontsize=15)
plt.ylabel('Count', fontsize=15)
plt.title('Residual histogram (data with noise and outliers)')
plt.show()

# ****************************
# Predict y value given x value

x_test = np.array([[14.0],])
y_result = reg.predict(x_test)

print(f'When x = {x_test[0]}, y = {y_result[0]}')