import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# data
alx_values = np.array([30,1])
sbp_values = np.array([30,1])
dbp_values = np.array([30,1])

X =alx_values.reshape(-1,1) # turns your 1-D ALX array into a column vector.

# sbp model 
model_sbp = LinearRegression().fit(X,sbp_values)
a_sbp     = model_sbp.coef_[0]
b_sbp     = model_sbp.intercept_
y_sbp_pred= model_sbp.predict(X)
r2_sbp    = r2_score(sbp_values, y_sbp_pred)

print(f"SBP  = {a_sbp:.4f} * ALX + {b_sbp:.4f}    (R² = {r2_sbp:.3f})")

# DBP model
model_dbp = LinearRegression().fit(X, dbp_values)
a_dbp     = model_dbp.coef_[0]
b_dbp     = model_dbp.intercept_
y_dbp_pred= model_dbp.predict(X)
r2_dbp    = r2_score(dbp_values, y_dbp_pred)

print(f"DBP  = {a_dbp:.4f} * ALX + {b_dbp:.4f}    (R² = {r2_dbp:.3f})")
