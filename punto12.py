import numpy as np
import matplotlib.pyplot as plt

#Datos:
x=[-50,-70,-180,-200,-240,-300,-330] #presión (P)
y=[1,2,3,4,5,6,7] #indice de refracción (n)

n = len(x)
x = np.array(x)
y = np.array(y)
sumx = sum(x)
sumy = sum(y)
sumx2 = sum(x*x)
sumy2 = sum(y*y)
sumxy = sum(x*y)
promx = sumx/n
promy = sumy/n
xint=np.array([0, -400])

m = (sumx*sumy - n*sumxy)/(sumx**2 - n*sumx2)
b = promy - m*promx
#b=0
m, b

xi=np.array([-0,-50,-70,-180,-200,-240,-300,-330,-400]) #P
yi=np.array([0,1,2,3,4,5,6,7]) #n

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

# Major ticks every 20, minor ticks every 5
major_ticks = np.arange(0, 8, 1)
minor_ticks = np.arange(0, -400, -10)

#ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)

#ax.grid(which='both')

ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.5)

plt.plot(x, y, 'o', label='Mediciones experimentales', color="black")
plt.plot(xi, m*xi + b, label='Predicción por ajuste lineal', color="green")
plt.xlabel('Presión manométrica (P) en milibares')
plt.ylabel('Número de franjas (m)')
#plt.title('Regresión lineal')
#plt.grid()
plt.legend(loc=4)

plt.xlim([0, -400])
plt.ylim([0, 8])
plt.show()

#Cálculo del el coeficiente de determinación (R2)
#Interpretación: Si R=1.0, entonces el modelo lineal predice con perfecta precisión. Si R=0.0, el ajuste lineal no predice con presición en absoluto.

sigmax = np.sqrt(sumx2/n - promx**2)
sigmay = np.sqrt(sumy2/n - promy**2)
sigmaxy = sumxy/n - promx*promy
R2 = (sigmaxy/(sigmax*sigmay))**2
R2