#funcion que calcula la distancia entre un cuerpo y todos los demas
import numpy as np
import numba as nm
from numba import jit
c=1.49e11 #distancia de la tierra al sol en metros
G=6.67e-11 #constante de gravitacion universal
m=1.989e30 #masa del sol en kg
n=9 #numero de planetas

@jit(nopython=True)
def distancia(x,y,i,j):
    distancia=np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
    return distancia


#función que calcula la fuerza en x
@jit(nopython=True)
def fuerza_x(x,y,M,f_x):
    for i in range(0,len(x)):
        suma=0.0
        for j in range(0,len(x)):
            if i!=j:
                suma=suma-M[j]*(x[i]-x[j])/distancia(x,y,i,j)**3
        f_x[i]=suma
    return f_x
    
#función que calcula la fuerza en y
@jit(nopython=True)
def fuerza_y(x,y,M,f_y):
    for i in range(0,len(x)):
        suma=0.0
        for j in range(0,len(x)):
            if i!=j:
                suma=suma-M[j]*(y[i]-y[j])/distancia(x,y,i,j)**3
        f_y[i]=suma
    return f_y

#funciones que calculan las posiciones nuevas en x e y
@jit(nopython=True)
def posnueva_x(x,vx,f_x,d,i,c,h):
    if d<778.5e9/c:
        return x[i]+h*vx[i]+0.5*h**2*f_x[i]
    if d>=778.5e9/c:
        hr=8*h
        return x[i]+hr*vx[i]+0.5*hr**2*f_x[i]

@jit(nopython=True)
def posnueva_y(y,vy,f_y,d,i,c,h):
    if d<778.5e9/c:
        return y[i]+h*vy[i]+0.5*h**2*f_y[i]
    if d>=778.5e9/c:
        hr=8*h
        return y[i]+hr*vy[i]+0.5*hr**2*f_y[i]

#funciones que calculas las nuevas w
@jit(nopython=True)    
def w_nueva_x(vx,f_x,d,i,c,h):
    if d<778.5e9/c:
        return vx[i]+0.5*h*f_x[i]
    if d>=778.5e9/c:
        hr=8*h
        return vx[i]+0.5*hr*f_x[i]

@jit(nopython=True)
def w_nueva_y(vy,f_y,d,i,c,h):
    if d<778.5e9/c:
        return vy[i]+0.5*h*f_y[i]
    if d>=778.5e9/c:
        hr=8*h
        return vy[i]+0.5*hr*f_y[i]

#funciones que claculan las nuevas velocidades
@jit(nopython=True)
def vel_nueva_x(w_x,f_x,d,i,c,h):
    if d<778.5e9/c:
        return w_x[i]+0.5*h*f_x[i]
    if d>=778.5e9/c:
        hr=8*h
        return w_x[i]+0.5*hr*f_x[i]
@jit(nopython=True)
def vel_nueva_y(w_y,f_y,d,i,c,h):
    if d<778.5e9/c:
        return w_y[i]+0.5*h*f_y[i]
    if d>=778.5e9/c:
        hr=8*h
        return w_y[i]+0.5*hr*f_y[i]

#funciones que calculan la energía cinética y potencial
@jit(nopython=True)
def Ec(vx,vy,M,T):
    for i in range(0,len(vx)):
        T[i]=0.5*M[i]*(vx[i]**2+vy[i]**2)
    return T

@jit(nopython=True)
def Ep(x,y,M,V):
    for i in range(0,len(x)):
        suma=0.0
        for j in range(0,len(x)):
            if i!=j:
                suma=suma-M[i]*M[j]/distancia(x,y,i,j)

        V[i]=suma
    return V 
 
#Función que calcula el momento angular
@jit(nopython=True)
def momento_angular(x,y,vx,vy,M,L):
    for i in range(len(x)):
        L[i]= distancia(x,y,i,0)*M[i]*np.sqrt(vx[i]**2+vy[i]**2) #Suma del momento angular de cada cuerpo
    return L

@jit(nopython=True)
def Energía_total(E):
    for i in range(0,len(E)):
        ET=np.sum(E)
    return ET

#funcion que calcula el momento angular total
@jit(nopython=True)
def momento_angular_total(L):
    for i in range(len(L)):
        LT=np.sum(L)
    return LT

#Calculo de la excentricidad
def excentricidad(m,E,L):
    e=np.zeros(n)
    for i in range(n):
        e[i]=np.sqrt(1+2*ETv[i]*LTv[i]**2/((m[i])**3))
    return e