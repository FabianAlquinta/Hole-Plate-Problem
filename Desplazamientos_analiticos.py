#Definiendo una función seno que reciba un valor en grados
def sen(grados):
    return math.sin(math.radians(grados))

#Definiendo una función coseno que reciba un valor en grados
def cos(grados):
    return math.cos(math.radians(grados))

#función que elimina la notación cientifica (opcional)
def format_float(num):
    return np.format_float_positional(num, trim='-')

#Recibe el numero de elemento y una lista vacia, para retornar una lista con los 4 nodos que le corresponden al elemento
def nodosdeelemento(element,lineaelemento):
    #El contador cuenta la cantidad de caracteres con espacio vacio en cada linea
    contador = 0
    nodo1 = ""
    nodo2 = ""
    nodo3 = ""
    nodo4 = ""
    #parametro para trabajar cualquier tipo de malla
    parametro_inf = int(lineas[1]) + 4
    for caracter in lineas[parametro_inf+element]:
        if caracter == " ":
            contador += 1
        #Luego de 5 caracteres en blanco, empiezan los valores utiles
        if contador == 5:
            #Se descarta el quinto caracter de espacio
            if caracter != " ":
                nodo1 += caracter
        if contador ==6 :
            if caracter != " ":
                nodo2 += caracter
        if contador == 7:
            if caracter != " ":
                nodo3 += caracter
        if contador == 8:
            if caracter != " ":
                nodo4 += caracter
    #Se agregan todos a una lista y se transforman de string a enteros
    lineaelemento.append(int(nodo1))
    lineaelemento.append(int(nodo2))
    lineaelemento.append(int(nodo3))
    lineaelemento.append(int(nodo4.rstrip('\n')))
    return lineaelemento

#Recibe lista con los nodos de algun elemento más 2 listas vacias (cx,cy) y retorna estas con las 4 coordenadas de x e y respectivamente.
def coordinates(lineaelemento,cx,cy):
    #El contador cuenta el primer caracter de espacio
    coordenadax = ""
    coordenaday = ""
    contador = 0
    for j in range(1,5):
        for caracter in lineas[1 + lineaelemento[j-1]]:
            if caracter == " ":
                contador += 1
            if contador == 1:
                #descartando el primer espacio
                if caracter != " ":
                    coordenadax += caracter
            if contador == 2:
                if caracter != " ":
                    coordenaday += caracter
        #Se añaden las coordenadas a las listas cx y cy como valores flotantes            
        cx.append(float(coordenadax))
        cy.append(float(coordenaday))
        coordenadax = ""
        coordenaday = ""
        contador = 0
    return cx,cy

#Función que reemplaza las coordenadas de x e y en la formula analítica y retorna 2 listas donde se almacenan los 4 desplazamientos en x e y respectivamente.
def resultados(Rpor4X,Rpor4Y,cx,cy):
    #Definiendo valores constantes
    E = 7000 #kPA
    poisson = 0.3
    T = 700 #kPA
    mu = (E)/(2*(1+poisson))
    k = (3-poisson)/(1+poisson)
    ro = 0.5 
    #ciclo for itera 4 veces ya que los elementos son de 4 nodos
    for i in range(4):
        x = cx[i] 
        y = cy[i]
        #aplicando coordenadas polares
        r = math.sqrt(x**2+y**2)
        teta = (math.atan2(y,x) * 180)/ math.pi # notese que se pasaron a grados
        uxx = (T/(4*mu))*( ((k+1)/2)*r*cos(teta) + ((ro**2)/r)*( (k+1)*cos(teta) + cos(3*teta) ) - ((ro**4)/(r**3))*cos(3*teta)   )
        uyy = (T/(4*mu))*( ((k-3)/2)*r*sen(teta) + ((ro**2)/r)*( (k-1)*sen(teta) + sen(3*teta) ) - ((ro**4)/(r**3))*sen(3*teta)   )
        Rpor4X.append(uxx)
        Rpor4Y.append(uyy)
    return Rpor4X,Rpor4Y

#Esta función va almacenando los resultados de desplazamiento en listas globales de desplazamiento en x e y hechas de 0, 
#y a medida que se van calculando, se van agregando a estas listas globales que almacenan cada valor en orden
def listascompletas(cx,cy,lineaelemento,listaux_pernodo,listauy_pernodo,listauz_pernodo):

    Rpor4X = []
    Rpor4Y = []
    resultados(Rpor4X,Rpor4Y,cx,cy) #se llenan las listas Rpor4X y Rpor4Y
    for i in range(4):
        listaux_pernodo[ lineaelemento[i] -1] = Rpor4X[i]
        listauy_pernodo[ lineaelemento[i] -1] = Rpor4Y[i]
        
    return listaux_pernodo,listauy_pernodo,listauz_pernodo

#Se crea la funcíon que hace un ciclo iterativo para cada elemento (utilizando las funciones anteriormente definidas),
# y retorna una lista con 3 listas dentro, las que corresponden a todos los desplzamientos en x, y, z.
def Desplazamientos():
    total = []
    parametro_inf = int(lineas[1]) + 4
    otro= int(lineas[parametro_inf]) + 1
    
    for ele in range(1,otro):
        
        lineaelemento = []
        nodosdeelemento(ele,lineaelemento) # se llenan lineaelemento con 4 nodos correspondientes al primer elemento
        
        cx = []
        cy = []
        coordinates(lineaelemento,cx,cy) #Se llenan listas de coordenadas cx y cy
        
        listascompletas(cx,cy,lineaelemento,listaux_pernodo,listauy_pernodo,listauz_pernodo)
    
    total.append(listaux_pernodo)
    total.append(listauy_pernodo)
    total.append(listauz_pernodo)
    return total

#Funcion que realiza el formato paraview a partir de la malla del Gmsh y los resultados obtenidos
def paraview_analitica():
    #creando archivo con formato vtk
    palabra="_analitica_desplazamiento.vtk"
    todo = nombre+palabra
    file = open(todo,"w")
    file.write("# vtk DataFile Version 1.0\n")
    file.write("MESH\n")
    file.write("ASCII\n\n")
    file.write("DATASET POLYDATA\n")
    nodos=int(lineas[1])
    file.write("POINTS  %i float" % (nodos))
    file.write("\n\n")

    archivito=open(nombre_malla,"r")
    lista=archivito.read().split(" ")
    c=0
    cuantas=0
    final=0
    #Imprimiendo las coordenadas de cada nodo en orden.
    for i in lista:
        final+=1
        c+=1
        if c==2:
            file.write(i)
        if c==3:
            file.write(" "+str(i)+" "+"0 \n")
            cuantas+=1
            c=0
        if cuantas==int(lineas[1]):
            break


    parametro_inf = int(lineas[1]) + 4
    otro= int(lineas[parametro_inf]) + 1
    elementos=int(lineas[parametro_inf])
    NN = elementos + elementos*4
    file.write("\n")
    file.write("POLYGONS  %i %i" % (nodos,NN) )
    file.write("\n")
    # Imprimiendo los nodos de cada elemento en filas
    for ele in range(1,otro):
        
        lineaelemento = []
        nodosdeelemento(ele,lineaelemento)
        x1 = lineaelemento[0]-1
        x2 = lineaelemento[1]-1
        x3 = lineaelemento[2]-1
        x4 = lineaelemento[3]-1

        
        file.write("4 "+str(x1)+" "+str(x2)+" "+str(x3)+" "+str(x4)+" \n")
        
    file.write("\n")
    file.write("POINT_DATA %i" % (nodos))
    file.write("\n")
    file.write("VECTORS Displacement float")
    file.write("\n")
    #Imprimiendo los valores de desplazamientos en orden
    for i in range(nodos):
        uxi = format_float(ux[i])
        uyi = format_float(uy[i])
        
        if i != (nodos-1):
            file.write(str(uxi)+" "+str(uyi)+ " "+ str(uz[i])+" \n" )
        if i == (nodos-1):
            file.write(str(uxi)+" "+str(uyi)+ " "+ str(uz[i]) +" \n")

    file.close()
    return lista

import numpy as np  
import math
#Esta es la malla Gmsh
nombre = input("Ingrese nombre de la malla (sin formato):")
formato = ".txt"
nombre_malla = nombre+formato
Malla = open(nombre_malla,"r")
lineas = Malla.readlines()
Malla.close()

#Definiendo listas globales de dimensiones nodosx1 con ceros, para posteriormente llenarlas
listaux_pernodo = []
listauy_pernodo = []
listauz_pernodo = []

for i in range( int(lineas[1]) ):
    listaux_pernodo.append(0)
    listauy_pernodo.append(0)
    listauz_pernodo.append(0)

ux=Desplazamientos()[0]
uy=Desplazamientos()[1]
uz=Desplazamientos()[2]

#Inicializando el programa
paraview_analitica()