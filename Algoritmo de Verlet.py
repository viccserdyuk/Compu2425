import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numba as nm
import math
from numba import jit
import funciones as fc
from matplotlib.patches import Circle
from matplotlib.animation import PillowWriter

c=1.49e11 #distancia de la tierra al sol en metros
G=6.67e-11 #constante de gravitacion universal
m=1.989e30 #masa del sol en kg
n=9 #numero de planetas
cv=np.sqrt(G*m/c)
M=np.array([1.0,0.33e24/m,4.87e24/m,5.97e24/m,0.642e24/m,1898e24/m,568e24/m,86.8e24/m,102e24/m], dtype=np.float64) #masas del sol y de mercurio
x=np.array([0.0,57.9e9/c,108.2e9/c,149.6e9/c,228e9/c,778.5e9/c,1432e9/c,2867e9/c,4515e9/c],dtype=np.float64)#posiciones en el eje x
y=np.zeros(len(x),dtype=np.float64)#posiciones en el eje y
b=np.zeros(len(x),dtype=np.float64)
a=x.copy()

vx=np.zeros(len(x))
vy=np.array([0.0,47.4e3/cv,35.0e3/cv,29.8e3/cv,24.1e3/cv,13.1e3/cv,9.7e3/cv,6.8e3/cv,5.4e3/cv],dtype=np.float64)#velocidades en el eje y
f_x=np.zeros(len(x))
f_y=np.zeros(len(x))
f_x=fc.fuerza_x(x,y,M,np.zeros(len(x)))#inciializamos las fuerzas en x
f_y=fc.fuerza_y(x,y,M,np.zeros(len(x)))#inicializamos las fuerzas en y

w_x=np.zeros(len(x))    
w_y=np.zeros(len(x))  

#Energía cinética, potencial y momento angular
T=np.zeros(len(x),dtype=np.float64)
V=np.zeros(len(x),dtype=np.float64)  
L=np.zeros(len(x),dtype=np.float64)
E=np.zeros(len(x),dtype=np.float64)
T=fc.Ec(vx,vy,M,np.zeros(len(x)))
V=fc.Ep(x,y,M,np.zeros(len(x)))
L=fc.momento_angular(x,y,vx,vy,M,np.zeros(len(x)))
E=T+V

t=0.0
h=0.01
t_c=np.sqrt(G*m/c**3)*t
tf=189.216
P_x=np.empty([0,len(x)])
P_y=np.empty([0,len(x)])
Em=np.empty([0,len(x)])
Lm=np.empty([0,len(x)])
LTv= []
ETv= []

#GIF
N = n  # Número de planetas
r = x # Distancia eje x al Sol (normalizadas)
# Frecuencias:
omega = vy #Velocidad angular
omega[0] = 0 #El sol no se mueve

T_gif = 400  # Pasos en el tiempo
t_gif = np.linspace(0, 13*np.pi, T_gif)

data = np.zeros((T_gif, N, 2))  # Crea el array de ceros, para el número de pasos en el tiempo, número de planetas y coordenadas (x,y)
data[:,:,0] = r[np.newaxis,:]*np.cos(omega[np.newaxis,:]*t_gif[:,np.newaxis]) #Posiciones en el eje X 
data[:,:,1] = r[np.newaxis,:]*np.sin(omega[np.newaxis,:]*t_gif[:,np.newaxis]) #Posiciones en el eje Y

# Arreglo para almacenar el tiempo en que cada planeta cruza a y negativo.
# Usamos -1 para indicar que aún no se ha registrado el cruce.
tiempos_cruce = [-1.0] * len(x)

while t_c < tf:
    P_x = np.vstack([P_x, x])
    P_y = np.vstack([P_y, y])
    Em=np.vstack([Em,E])
    Lm=np.vstack([Lm,L])
    LT=fc.momento_angular_total(L)
    LTv.append(LT)
    ET= fc.Energía_total(E)
    ETv.append(ET)
    
    for k in range(len(x)):
        d = fc.distancia(x, y, k, 0)
        w_x[k] = fc.w_nueva_x(vx, f_x, d, k, c, h)
        w_y[k] = fc.w_nueva_y(vy, f_y, d, k, c, h)
        x[k] = fc.posnueva_x(x, vx, f_x, d, k, c, h)
        y[k] = fc.posnueva_y(y, vy, f_y, d, k, c, h)
        f_x = fc.fuerza_x(x, y, M, f_x)
        f_y = fc.fuerza_y(x, y, M, f_y)
        vx[k] = fc.vel_nueva_x(w_x, f_x, d, k, c, h)
        vy[k] = fc.vel_nueva_y(w_y, f_y, d, k, c, h)
        L=fc.momento_angular(x,y,vx,vy,M,L)
        T=fc.Ec(vx,vy,M,T)
        V=fc.Ep(x,y,M,V)
        E=T+V
        # Si la coordenada y se vuelve negativa y aún no se ha registrado el tiempo, lo guardamos.
        if y[k] < 0 and tiempos_cruce[k] == -1.0:
            tiempos_cruce[k] = t_c
        #Vector para almacenar el eje y
        if abs(x[k]) < 1e-2:
         b[k]=y[k]
            
    t_c += h

ETv=np.array(ETv)
LTv=np.array(LTv)
e=np.sqrt(1-b[4]**2/a[4]**2) #Excentricidad
print(e)

"""a=np.sqrt(1+(2*E[7]*L[7]**2)/(G**2*M[0]**2*M[7]**3))*G
print(a)"""

labels=["Sol", "Mercurio", "Venus", "Tierra", "Marte", "Jupiter", "Saturno", "Urano", "Neptuno" ] #etiquetamos los planetas
for n in range(0,len(x)):
    plt.plot(P_x[:,n],P_y[:,n],label=labels[n])

plt.axis("equal")
plt.xlabel("Posición X (AU)")
plt.ylabel("Posición Y (AU)")
plt.title("Orbitas planetas")
plt.legend()
plt.show()

"""
#Para la creación del GIF
def update_plot(frame):
    
    plt.cla()  # Limpia el anterior ploteo
    
    for i in range(N):
        
        plt.plot(data[0:frame+1, i, 0], 
                 data[:frame+1, i, 1], 
                 color=plt.cm.viridis(i/N), alpha=0.3, linewidth=2)  # Plot previous positions with shading
        marker_size = 3 if i <= 5 else 8  # Tamaño según el índice del planeta
        plt.plot(data[frame, i, 0], data[frame, i, 1], 'o', color=plt.cm.viridis(i/N), markersize=marker_size)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.axis('off')  # Turn off axes

fig, ax = plt.subplots()
ani = animation.FuncAnimation(fig, update_plot, frames=T_gif, interval=50)
ani.save('planets_movement.gif', writer='pillow')
plt.show()
"""

#Calculo del periodo orbital de cada planeta
print("Periodos orbitales para Mercurio hasta Marte:")
for i in range(1, 5):  # índices 1:Mercurio, 2:Venus, 3:Tierra, 4:Marte
    if tiempos_cruce[i] != -1.0:
        # Se asume que el cruce corresponde a la mitad del período; se multiplica por 2.
        periodo_dias = 2 * np.pi * tiempos_cruce[i] * 1.58e6 / 86400
        print(f"{labels[i]}: {periodo_dias:.2f} días")

print("\nPeriodos orbitales para Jupiter hasta Neptuno:")
for i in range(5, len(labels)):
    if tiempos_cruce[i] != -1.0:
        periodo_dias = 2 * 8 * np.pi * tiempos_cruce[i] * 1.58e6 / 86400
        print(f"{labels[i]}: {periodo_dias:.2f} días")