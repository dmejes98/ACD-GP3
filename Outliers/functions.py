from scipy.spatial.distance import mahalanobis as mhl
from sklearn.covariance import LedoitWolf as lw
import numpy as np

def outliers (df, lam, cv):
  if cv: cov = df.cov()
  else: cov = list(lw().fit(df).covariance_) + lam* np.eye(2)
  # else: cov = list(np.cov(df))
  
  inv_cov = np.linalg.pinv(cov)
  print('Condición ', np.linalg.cond(cov))
  median = [np.median(df['Volume_est']), np.median(df['rdto'])]
  mahal_distances = []
  for i in range(len(df)):
    mahal_distances.append(mhl(median, list(df.iloc[i]), inv_cov))

  return mahal_distances

## Función para filtrado de outliers utilizando la distancia de Mahalanobis
def cov1 (m, n):
  cov = m.cov() + n*np.eye(2)
  cond = np.linalg.cond(cov)
  det = np.linalg.det(cov)

  return cov, cond, det
