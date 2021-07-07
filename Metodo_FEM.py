#Recibe numero de gauss, y listas litz y listw vacias, para retornar estas ultimas llenas con los valores de peso y variables corresponientes
def numbers(numberofpoints,listz,listw):
    while numberofpoints >= 1:
        if numberofpoints == 1:
            w = 2
            z = 0
            listz.append(z)
            listw.append(w)
            return listz, listw
        if numberofpoints == 2:
            w1 = 1
            w2 = 1
            z1 = -0.5773502692
            z2 = 0.5773502692
            listz.append(z1)
            listz.append(z2)
            listw.append(w1)
            listw.append(w2)
            return listz, listw
        elif numberofpoints == 3:
            w1 = 0.55555
            w2 = 0.88888
            w3 = 0.55555
            z1 = -0.7745966692
            z2 = 0
            z3 = 0.7745966692
            listz.append(z1)
            listz.append(z2)
            listz.append(z3)
            listw.append(w1)
            listw.append(w2)
            listw.append(w3)
            return listz, listw
        elif numberofpoints == 4:
            w1 = 0.3478548451
            w2 = 0.6521451549
            w3 = 0.6521451549
            w4 = 0.3478548451
            z1 = -0.8611363116
            z2 = -0.3399810436
            z3 = 0.3399810436
            z4 = 0.8611363116
            listz.append(z1)
            listz.append(z2)
            listz.append(z3)
            listz.append(z4)
            listw.append(w1)
            listw.append(w2)
            listw.append(w3)
            listw.append(w4)
            return listz, listw
        elif numberofpoints == 5:
            w1 = 0.2369268851
            w2 = 0.4786286705
            w3 = 0.56888
            w4 = 0.4786286705
            w5 = 0.2369268851
            z1 = -0.9061798459
            z2 = -0.5384693101
            z3 = 0
            z4 = 0.5384693101
            z5 = 0.9061798459
            listz.append(z1)
            listz.append(z2)
            listz.append(z3)
            listz.append(z4)
            listz.append(z5)
            listw.append(w1)
            listw.append(w2)
            listw.append(w3)
            listw.append(w4)
            listw.append(w5)
            return listz, listw
        elif numberofpoints == 6:
            w1 = 0.1713244924
            w2 = 0.3607615730
            w3 = 0.4679139346
            w4 = 0.4679139346
            w5 = 0.3607615730
            w6 = 0.1713244924
            z1 = -0.9324695142
            z2 = -0.6612093865
            z3 = -0.2386191861
            z4 = 0.2386191861
            z5 = 0.6612093865
            z6 = 0.9324695142
            listz.append(z1)
            listz.append(z2)
            listz.append(z3)
            listz.append(z4)
            listz.append(z5)
            listz.append(z6)
            listw.append(w1)
            listw.append(w2)
            listw.append(w3)
            listw.append(w4)
            listw.append(w5)
            listw.append(w6)
            return listz, listw

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
    for caracter in lineas[parametro_inf + element]:
        if caracter == " ":
            contador += 1
        #Luego de 5 caracteres en blanco, empiezan los valores utiles
        if contador == 5:
            #Se descarta el quinto caracter de espacio
            if caracter != " ":
                nodo1 += caracter
        if contador == 6 :
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
    return cx, cy  #Se agrego return aca

#Calcula el determinante del jacobiano
def JacobianDet(cx,cy,eta,xi):
    x1 = cx[0]
    x2 = cx[1]
    x3 = cx[2]
    x4 = cx[3]
    y1 = cy[0]
    y2 = cy[1]
    y3 = cy[2]
    y4 = cy[3]
    dxdxi = ((1-eta)/4) * (x2-x1) + ((1+eta)/4) * (x3-x4)
    dydxi = ((1-eta)/4) * (y2-y1) + ((1+eta)/4) * (y3-y4)
    dxdeta = ((1+xi)/4) * (x3-x2) + ((1-xi)/4) * (x4-x1)
    dydeta = ((1+xi)/4) * (y3-y2) + ((1-xi)/4) * (y4-y1)
    detj = dxdxi*dydeta - dydxi*dxdeta 
    return detj

#Calcula la matriz del Jacobiano
def JacobianMatrix(cx,cy,eta,xi):
    MatrixJacob = []
    row1 = []
    row2 = []
    
    x1 = cx[0]
    x2 = cx[1]
    x3 = cx[2]
    x4 = cx[3]
    y1 = cy[0]
    y2 = cy[1]
    y3 = cy[2]
    y4 = cy[3]
    
    dxdxi = ((1-eta)/4) * (x2-x1) + ((1+eta)/4) * (x3-x4)
    dydxi = ((1-eta)/4) * (y2-y1) + ((1+eta)/4) * (y3-y4)
    dxdeta = ((1+xi)/4) * (x3-x2) + ((1-xi)/4) * (x4-x1) 
    dydeta = ((1+xi)/4) * (y3-y2) + ((1-xi)/4) * (y4-y1) 
    
    row1.append(dxdxi)
    row1.append(dydxi)
    row2.append(dxdeta)
    row2.append(dydeta)
    MatrixJacob.append(row1)
    MatrixJacob.append(row2)
    array_MatrixJacob = np.asarray(MatrixJacob)
    return array_MatrixJacob

#Devuelve una lista arbitraria (de tamaño 8) que contiene las derivadas de N c/r a eta y xi
def dNdxi_dNdeta(eta,xi):
    calculadas = []
    dN1dxi = (1/4) *(eta-1)
    dN1deta = (1/4) *(xi-1)
    dN2dxi = (1/4) *(1-eta)
    dN2deta = (1/4) *(-xi-1)
    dN3dxi = (1/4) *(eta+1)
    dN3deta = (1/4) *(xi+1)
    dN4dxi = (1/4) *(-eta-1)
    dN4deta = (1/4) *(1-xi)
    calculadas.append(dN1dxi)
    calculadas.append(dN1deta)
    calculadas.append(dN2dxi)
    calculadas.append(dN2deta)
    calculadas.append(dN3dxi)
    calculadas.append(dN3deta)
    calculadas.append(dN4dxi)
    calculadas.append(dN4deta)
    return calculadas

#Recibe funcion dNdxi_dNdeta(eta,xi) y retorna un arreglo con 4 matrices de 2x2 dentro para poder operar matematicamente
def ComponentsdNdX_dNdY(MatrixdN):
    Allcomponents = []
    result1 = []
    for i in range(2):
        temp = []
        temp.append(MatrixdN[i])
        result1.append(temp)
    result2 = []
    for i in range(2,4):
        temp = []
        temp.append(MatrixdN[i])
        result2.append(temp)
    result3 = []
    for i in range(4,6):
        temp = []
        temp.append(MatrixdN[i])
        result3.append(temp)
    result4 = []
    for i in range(6,8):
        temp = []
        temp.append(MatrixdN[i])
        result4.append(temp)
    Allcomponents.append(result1)
    Allcomponents.append(result2)
    Allcomponents.append(result3)
    Allcomponents.append(result4)
    array_Allcomponents = np.asarray(Allcomponents)
    return array_Allcomponents

#Calcula las derivadas de N con respecto a x e y, y las retorna en una lista (de tamaño 8) con los resultados de cada una.
def dNdX_dNdY(Matrixjaco,eta,xi):
    Allrp = []
    lista = []
    #calculando inversa del jacobiano
    JacobianMatrix_inv = np.linalg.inv(Matrixjaco)
    p = ComponentsdNdX_dNdY(dNdxi_dNdeta(eta,xi))
    p1 = p[0]
    p2 = p[1]
    p3 = p[2]
    p4 = p[3]
    
    r1 = np.dot(JacobianMatrix_inv,p1)
    r2 = np.dot(JacobianMatrix_inv,p2)
    r3 = np.dot(JacobianMatrix_inv,p3)
    r4 = np.dot(JacobianMatrix_inv,p4)
    
    Allrp.append(r1)
    Allrp.append(r2)
    Allrp.append(r3)
    Allrp.append(r4)
    array_Allrp = np.asarray(Allrp)
    lista = array_Allrp.tolist()
    return lista

#Utilizando los resultados obtenidos de dNdX_dNdY(Matrixjaco,eta,xi), da forma a la matriz [B] y la retorna como array
def B_matrix(line):
    B = []
    fila1 = []
    for i in range(4):
        fila1.append(line[i][0][0])
        fila1.append(0)
    fila2 = []
    for i in range(4):
        fila2.append(0)
        fila2.append(line[i][1][0])
    fila3 = []
    for i in range(4):
        fila3.append(line[i][0][0])
        fila3.append(line[i][1][0])
    B.append(fila1)
    B.append(fila2)
    B.append(fila3)
    array_B = np.asarray(B)
    return array_B
#Matriz de esfuerzo plano
def D_Matrix():
    E = 7000 
    poisson = 0.3
    D = [  [1,poisson,0] , [poisson,1,0] , [0,0, (1-poisson)/2]     ]
    array_D = np.asarray(D)
    factor = (E)/ (1-(poisson**2))     
    return factor*array_D

#Aplica cuadratura gaussiana a cada elemento para calcular las matrices de rigidez locales [k]
def sumatoria_klocal(n,cx,cy,listz,listw):
    total = []
    i = 1
    while i <= n:
        j = 1
        while j <= n:
            B_transp = np.transpose( B_matrix( dNdX_dNdY(JacobianMatrix(cx,cy,listz[i-1],listz[i-1]),listz[i-1],listz[i-1]) )  )
            B_normal = B_matrix( dNdX_dNdY(JacobianMatrix(cx,cy,listz[i-1],listz[i-1]),listz[i-1],listz[i-1]) )
            detjaco = JacobianDet(cx,cy,listz[i-1],listz[i-1])
            
            total.append( np.dot(np.dot(B_transp,D_Matrix()), B_normal) * detjaco * listw[i-1] * listw[j-1])
            j += 1
        i += 1
    return sum(total)

#Recibe una matriz k local, la descompone en matrices 2x2 y las retorna
def All_k(single_matrix,green,red,blue,center):
    a = single_matrix
    #Nota: los nombres de colores se utilizan debido a que antes de programarlo, fue necesario hacer un dibujo a color descomponiendo la matriz k local
    # y luego para acoplar en la k global de dimensiones (2*nodos)
    green_array1 = np.array(      [ [a[0][0],a[0][1]] , [a[1][0],a[1][1]] ]    )
    green_array2 = np.array(      [ [a[0][6],a[0][7]] , [a[1][6],a[1][7]] ]    )
    green_array3 = np.array(      [ [a[6][0],a[6][1]] , [a[7][0],a[7][1]] ]    )
    green_array4 = np.array(      [ [a[6][6],a[6][7]] , [a[7][6],a[7][7]] ]    )
    
    red_array1 = np.array(        [ [a[2][0],a[2][1]] , [a[3][0],a[3][1]] ]    )
    red_array2 = np.array(        [ [a[2][6],a[2][7]] , [a[3][6],a[3][7]] ]    )
    red_array3 = np.array(        [ [a[4][0],a[4][1]] , [a[5][0],a[5][1]] ]    )
    red_array4 = np.array(        [ [a[4][6],a[4][7]] , [a[5][6],a[5][7]] ]    )
    
    blue_array1 = np.array(       [ [a[0][2],a[0][3]] , [a[1][2],a[1][3]] ]    )
    blue_array2 = np.array(       [ [a[0][4],a[0][5]] , [a[1][4],a[1][5]] ]    )
    blue_array3 = np.array(       [ [a[6][2],a[6][3]] , [a[7][2],a[7][3]] ]    )
    blue_array4 = np.array(       [ [a[6][4],a[6][5]] , [a[7][4],a[7][5]] ]    )
    
    center_array1 = np.array(     [ [a[2][2],a[2][3]] , [a[3][2],a[3][3]] ]    )
    center_array2 = np.array(     [ [a[2][4],a[2][5]] , [a[3][4],a[3][5]] ]    )
    center_array3 = np.array(     [ [a[4][2],a[4][3]] , [a[5][2],a[5][3]] ]    )
    center_array4 = np.array(     [ [a[4][4],a[4][5]] , [a[5][4],a[5][5]] ]    )
    
    green.append(green_array1)
    green.append(green_array2)
    green.append(green_array3)
    green.append(green_array4)
    
    red.append(red_array1)
    red.append(red_array2)
    red.append(red_array3)
    red.append(red_array4)
    
    blue.append(blue_array1)
    blue.append(blue_array2)
    blue.append(blue_array3)
    blue.append(blue_array4)
    
    center.append(center_array1)
    center.append(center_array2)
    center.append(center_array3)
    center.append(center_array4)
    
    return green, red, blue, center

#se define la matriz k global con ceros y de dimension 2xnodos
def kglobal_empty():
    nodes = int(lineas[1])
    empty_global = []
    for i in range(nodes*2):
        empty_global.append([])
        for j in range(nodes*2):
            empty_global[i].append(0.0)
            
    arr_empty_global = np.asarray(empty_global)
    return arr_empty_global

#Recibe la descomposicion de una matriz k local y va sumando cada parte donde le corresponda en la matriz k global
def assemble_k(list_nodes,global_matrix,green,red,blue,center):
    node1 = int(list_nodes[0])
    node2 = int(list_nodes[1])
    node3 = int(list_nodes[2])
    node4 = int(list_nodes[3])
    #Green Matrix
    for k in range(4):
        M_green  = green[k]
        for i in range(2):
            for j in range(2):
                if k == 0: #1
                    global_matrix[(node1-1)*2 + i][(node1-1)*2 + j] += M_green[i][j]
                if k == 1: #2
                    global_matrix[(node1-1)*2 + i][(node4-1)*2 + j] += M_green[i][j]
                if k == 2: #3
                    global_matrix[(node4-1)*2 + i][(node1-1)*2 + j] += M_green[i][j]
                if k == 3: #4
                    global_matrix[(node4-1)*2 + i][(node4-1)*2 + j] += M_green[i][j]
    #Red Matrix
    for k in range(4):
        M_red  = red[k]
        for i in range(2):
            for j in range(2):
                if k == 0: #1
                    global_matrix[(node2-1)*2 + i][(node1-1)*2 + j] += M_red[i][j]
                if k == 1: #2
                    global_matrix[(node2-1)*2 + i][(node4-1)*2 + j] += M_red[i][j]
                if k == 2: #3
                    global_matrix[(node3-1)*2 + i][(node1-1)*2 + j] += M_red[i][j]
                if k == 3: #4
                    global_matrix[(node3-1)*2 + i][(node4-1)*2 + j] += M_red[i][j]
    #Blue Matrix
    for k in range(4):
        M_blue  = blue[k]
        for i in range(2):
            for j in range(2):
                if k == 0: #1
                    global_matrix[(node1-1)*2 + i][(node2-1)*2 + j] += M_blue[i][j]
                if k == 1: #2
                    global_matrix[(node1-1)*2 + i][(node3-1)*2 + j] += M_blue[i][j]
                if k == 2: #3
                    global_matrix[(node4-1)*2 + i][(node2-1)*2 + j] += M_blue[i][j]
                if k == 3: #4
                    global_matrix[(node4-1)*2 + i][(node3-1)*2 + j] += M_blue[i][j]
    #Center Matrix
    for k in range(4):
        M_center  = center[k]
        for i in range(2):
            for j in range(2):
                if k == 0: #1
                    global_matrix[(node2-1)*2 + i][(node2-1)*2 + j] += M_center[i][j]
                if k == 1: #2
                    global_matrix[(node2-1)*2 + i][(node3-1)*2 + j] += M_center[i][j]
                if k == 2: #3
                    global_matrix[(node3-1)*2 + i][(node2-1)*2 + j] += M_center[i][j]
                if k == 3: #4
                    global_matrix[(node3-1)*2 + i][(node3-1)*2 + j] += M_center[i][j]
    return global_matrix

#Calcula la matriz k global de la malla
def calculoGauss(puntodegauss):
    #generalizando a todas las mallas
    parametro_inf = int(lineas[1]) + 4
    otro = int(lineas[parametro_inf]) + 1 #numero de elementos
    #se define matriz k global con ceros
    global_matrix = kglobal_empty()
    for ele in range(1,otro):
        listz = []
        listw = []
        #Definiendo valores de zi, wi
        numbers(puntodegauss,listz,listw)
        #Definiendo los nodos de cada elemento
        lineaelemento = []
        nodosdeelemento(ele,lineaelemento)
        #Guardando las coordenadas de cada nodo en cx,cy
        cx = []
        cy = []
        coordinates(lineaelemento,cx,cy)
        #Definiendo matrices que se llenaran con respecto al k local
        green = [] 
        red = []
        blue = []
        center = []
        klocal = sumatoria_klocal(puntodegauss,cx,cy,listz,listw)
        #llenando las listas green, red, blue, center
        All_k(klocal,green,red,blue,center)
        #Se acopla matriz k local a la matriz global matrix (k global)
        assemble_k(lineaelemento,global_matrix,green,red,blue,center)
        
    return global_matrix

############################################################################################

#                                    FUEZAS EXTERNAS

############################################################################################

#definiendo matriz N y retornandola con la forma necesaria
def N_etaxi(eta,xi):
    p = []
    N1 = ((1-xi)/(2)) * ((1-eta)/(2))
    N2 = ((1+xi)/(2)) * ((1-eta)/(2))
    N3 = ((1+xi)/(2)) * ((1+eta)/(2))
    N4 = ((1-xi)/(2)) * ((1+eta)/(2))
    p.append(N1)
    p.append(N2)
    p.append(N3)
    p.append(N4)
    lista = [ [p[0],0,p[1],0,p[2],0,p[3],0] , [0,p[0],0,p[1],0,p[2],0,p[3]] ]
    array_lista = np.asarray(lista)
    return array_lista

#Cuadratura gaussiana que retorna una matriz f local
def sumatoria_Nelocal(n,cx,cy,eta,xi,listw):
    total = []
    sigmax= 700
    t_lista = [[sigmax],[0]]
    t_array = np.asarray(t_lista)
    i = 1
    while i <= n:
        j = 1
        while j <= n:
            detjaco = JacobianDet(cx,cy,eta[i-1],xi[i-1])

            total.append(  np.dot(np.transpose(N_etaxi(eta[i-1],xi[i-1])),t_array) * detjaco * listw[i-1] * listw[j-1]  )
            j += 1
        i += 1
    return sum(total)

#Funcion que calcula la F local para cada elemento y si existe una fuerza aplicada en un lado del elemento, la calcula igualmente
#Y retorna cada F local en una lista continua
def calculoGauss_2(puntodegauss):
    vectoresF = []
    parametro_inf = int(lineas[1]) + 4
    otro = int(lineas[parametro_inf]) + 1
    for nod in range(1,otro):
        listz = []
        listw = []
        #Definiendo valores de zi, wi
        numbers(puntodegauss,listz,listw)
        #Definiendo los nodos de cada elemento
        lineaelemento = []
        nodosdeelemento(nod,lineaelemento)
        #Guardando las coordenadas de cada nodo en cx,cy
        cx = []
        cy = []
        coordinates(lineaelemento,cx,cy)
        #Sumando los resultados de la integral en cada elemento
        if (cx[0] == 2 and cx[1] == 2): #Fuerza en LADO 1
            list_eta = []
            list_xi = listz
            for i in range(puntodegauss):
                list_eta.append(-1)
            M = sumatoria_Nelocal(puntodegauss,cx,cy,list_eta,list_xi,listw)
            M.tolist()
            vectoresF.append(M)
            continue
        if (cx[1] == 2 and cx[2] == 2): #Fuerza en LADO 2
            list_xi = []
            list_eta = listz
            for i in range(puntodegauss):
                list_xi.append(1)
            M = sumatoria_Nelocal(puntodegauss,cx,cy,list_eta,list_xi,listw)
            M.tolist()
            vectoresF.append(M)
            continue
            
        if (cx[2] == 2 and cx[3] == 2): #Fuerza en LADO 3
            list_eta = []
            list_xi = listz
            for i in range(puntodegauss):
                list_eta.append(1)
            M = sumatoria_Nelocal(puntodegauss,cx,cy,list_eta,list_xi,listw)
            M.tolist()
            vectoresF.append(M)
            continue

        if (cx[3] == 2 and cx[0] == 2): #Fuerza en LADO 4
            list_xi = []
            list_eta = listz
            for i in range(puntodegauss):
                list_xi.append(-1)
            M = sumatoria_Nelocal(puntodegauss,cx,cy,list_eta,list_xi,listw)
            M.tolist()
            vectoresF.append(M)
            continue
            
        #SE AGREGARA MATRIZ CON 0 CUANDO NO ENTRE A NINGUN IF
        else:

            nulo =   [[0],[0],[0],[0],[0],[0],[0],[0]]
            vectoresF.append(nulo)

    array_vectoresF = np.asarray(vectoresF)
    return array_vectoresF

#Se define una f_global vacia de dimension (2*nodos x 1)
def fglobal_empty():
    nodes = int(lineas[1])
    empty_global = []
    
    for i in range(nodes*2):
        empty_global.append([])
        
        for j in range(1):
            empty_global[i].append(0.0)

    arr_empty_global = np.asarray(empty_global)
    return arr_empty_global

#Recibe una matriz F local, la descompone en matrices 2x1 y las retorna
def All_f(single_matrix,green,red,blue,center):
    a = single_matrix
    
    green_array = np.array(      [  [a[0][0]] , [a[1][0]]  ]  )
    red_array = np.array(        [  [a[2][0]] , [a[3][0]]  ]  )
    blue_array = np.array(       [  [a[4][0]] , [a[5][0]]  ]  )
    center_array = np.array(     [  [a[6][0]] , [a[7][0]]  ]  )
    
    green.append(green_array)
    red.append(red_array)
    blue.append(blue_array)
    center.append(center_array)
    
    return green, red, blue, center

#Recibe la descomposicion de una matriz f local y va sumando cada parte donde le corresponda en la matriz f global
def assemble_f(list_nodes,global_matrix,green,red,blue,center):
    node1 = int(list_nodes[0])
    node2 = int(list_nodes[1])
    node3 = int(list_nodes[2])
    node4 = int(list_nodes[3])
    #Green Matrix
    for i in range(2):
        global_matrix[ (node1-1)*2 + i ] += green[0][i][0]
    #Red Matrix
    for i in range(2):
        global_matrix[ (node2-1)*2 + i ] += red[0][i][0]
    #Blue Matrix
    for i in range(2):
        global_matrix[ (node3-1)*2 + i ] += blue[0][i][0]
    #Center Matrix
    for i in range(2):
        global_matrix[ (node4-1)*2 + i ] += center[0][i][0]

    return global_matrix

#Calcula la matriz F global de la malla
def calculoGauss_f(puntodegauss):
    #generalizando a todas las mallas
    parametro_inf = int(lineas[1]) + 4
    otro = int(lineas[parametro_inf]) + 1
    global_matrix = fglobal_empty()
    
    for ele in range(1,otro):
        listz = []
        listw = []
        #Definiendo valores de zi, wi
        numbers(puntodegauss,listz,listw)
        #Definiendo los nodos de cada elemento
        lineaelemento = []
        nodosdeelemento(ele,lineaelemento)
        #Guardando las coordenadas de cada nodo en cx,cy
        cx = []
        cy = []
        coordinates(lineaelemento,cx,cy)
        #Definiendo matrices que se llenaran con respecto al F local
        green = [] 
        red = []
        blue = []
        center = []
        flocales = calculoGauss_2(puntodegauss)
        flocal = flocales[ele-1]
        #llenando las listas green, red, blue, center
        All_f(flocal,green,red,blue,center)
        #Se acopla matriz F local a la matriz global matrix (f global)
        assemble_f(lineaelemento,global_matrix,green,red,blue,center)
        
    return global_matrix

############################################################################################

#                                  K MODIFICADA   (IMPONIENDO CONDICIONES DE BORDE)

############################################################################################

#Aplica la condicion de ux=0 en eje vertical
#Recibe indice y matriz k global para hacer fila y columna de ceros en la posicion dada, y un 1 en la posicion del indice
def row_column_ux(inde,matrix):
    index = (inde - 1)*2
    for i in range( matrix.__len__() ):
        matrix[index][i] = 0
        
    for i in matrix:
        i[index] = 0
    matrix[index][index] = 1
    return matrix

#Aplica la condicion de uy=0 en eje horizontal
#Recibe indice y matriz k global para hacer fila y columna de ceros en la posicion dada, y un 1 en la posicion del indice
def row_column_uy(inde,matrix):
    index = (inde)*2 - 1
    for i in range( matrix.__len__() ):
        matrix[index][i] = 0
        
    for i in matrix:
        i[index] = 0
    matrix[index][index] = 1
    return matrix

#Identifica los nodos que se encuentran en eje vertical y horizontal y los retorna en lista
def identificandonodoslimites(ambas):
    parametro_inf = int(lineas[1]) + 4
    otro= int(lineas[parametro_inf]) + 1
    nodosdeejey = []
    nodosdeejex = []
    for ele in range(1,otro):

        lineaelemento = []
        nodosdeelemento(ele,lineaelemento)
        #Guardando las coordenadas de cada nodo en cx,cy
        cx = []
        cy = []
        coordinates(lineaelemento,cx,cy)
        for j in range(4):
            if cx[j] == 0:
                nodosdeejey.append(lineaelemento[j])
        for j in range(4):
            if cy[j] == 0:
                nodosdeejex.append(lineaelemento[j])
    
    #Se revisa en cada elemento los nodos de los extremos y se agregan a una lista,
    #por lo que es necesario la funcion set(), para eliminar los nodos repetidos
    set_nodosdeejey = set(nodosdeejey)
    list_set_nodosdeejey = list(set_nodosdeejey)

    set_nodosdeejex = set(nodosdeejex)
    list_set_nodosdeejex = list(set_nodosdeejex)
    
    ambas.append(list_set_nodosdeejey)
    ambas.append(list_set_nodosdeejex)
    return ambas

#Se imponen las condiciones de desplazamiento a la matriz k global
def modificandok(kglobal):
    ambas = []
    identificandonodoslimites(ambas)
    #aplicando conciciones de borde en eje vertical
    ejey = ambas[0] #[4,5,9]
    for i in range(len(ejey)):
        nodes_ejey = ejey[i]
        row_column_ux(nodes_ejey,kglobal)
    #aplicando conciciones de borde en eje horizontal
    ejex = ambas[1] #[1,2,6]
    for j in range(len(ejex)):
        nodes_ejex = ejex[j]
        row_column_uy(nodes_ejex,kglobal)

    return kglobal

############################################################################################

#                                 F MODIFICADA   (IMPONIENDO CONDICIONES DE BORDE)

############################################################################################

#Aplica la condicion de ux=0 en eje vertical
#Recibe indice y matriz F global para hacer hacer 0 en la posicion del indice dado
def row_column_f_ux(inde,matrix):
    index = (inde - 1)*2
    for i in range( matrix.__len__() ):
        if i == index:
            matrix[index][0] = 0
    return matrix

#Aplica la condicion de uy=0 en eje vertical
#Recibe indice y matriz F global para hacer hacer 0 en la posicion del indice dado
def row_column_f_uy(inde,matrix):
    index = (inde)*2 - 1
    for i in range( matrix.__len__() ):
        if i == index:
            matrix[index][0] = 0
    return matrix

#Se imponen las condiciones de desplazamiento a la matriz F global
def modificandof(fglobal):
    ambas = []
    identificandonodoslimites(ambas)
    #aplicando conciciones de borde en eje vertical
    ejey = ambas[0] #[4,5,9]
    for i in range(len(ejey)):
        nodes_ejey = ejey[i]
        row_column_f_ux(nodes_ejey,fglobal)
    #aplicando conciciones de borde en eje horizontal
    ejex = ambas[1] #[1,2,6]
    for j in range(len(ejex)):
        nodes_ejex = ejex[j]
        row_column_f_uy(nodes_ejex,fglobal)
    return fglobal

############################################################################################

#                                DESPLAZAMIENTOS

############################################################################################

#Retorna matriz de desplazamientos
def displacement(gauss):
    #imponiendo las condiciones de borde en matriz k y f globales
    k_global_final = modificandok(calculoGauss(gauss))
    f_global_final = modificandof(calculoGauss_f(gauss))
    #calculando la inversa de la matriz k global
    k_inv = np.linalg.inv(  k_global_final      )
    disp = np.dot(k_inv,f_global_final)
    return disp

############################################################################################

#                                PARAVIEW PARA METODO FEM

############################################################################################
#Funcion que realiza el formato paraview a partir de la malla del Gmsh y los resultados de desplazamiento
def paraview_FEM(disp):
    palabra="_FEM.vtk"
    todo = nombre+palabra
    #creando archivo con formato vtk
    file = open(todo,"w")
    file.write("# vtk DataFile Version 1.0\n")
    file.write("MESH\n")
    file.write("ASCII\n\n")
    file.write("DATASET POLYDATA\n")
    nodos = int(lineas[1])
    file.write("POINTS  %i float" % (nodos))
    file.write("\n\n")

    archivito = open(nombre_malla,"r")
    lista = archivito.read().split(" ")
    c = 0
    cuantas = 0
    #Imprimiendo las coordenadas de cada nodo en orden.
    for i in lista:
        c+=1
        if c == 2:
            file.write(i)
        if c == 3:
            file.write(" "+str(i)+" "+"0 \n")
            cuantas+=1
            c = 0
        if cuantas == int(lineas[1]):
            break

    parametro_inf = int(lineas[1]) + 4
    otro = int(lineas[parametro_inf]) + 1
    elementos = int(lineas[parametro_inf])
    NN = elementos + elementos*4
    file.write("\n")
    file.write("POLYGONS  %i %i" % (nodos,NN) )
    file.write("\n")
    # Imprimiendo los nodos de cada elemento en filas
    for ele in range(1,otro):
        
        lineaelemento = []
        nodosdeelemento(ele,lineaelemento)
        x1 = lineaelemento[0] - 1
        x2 = lineaelemento[1] - 1
        x3 = lineaelemento[2] - 1
        x4 = lineaelemento[3] - 1

        file.write("4 "+str(x1)+" "+str(x2)+" "+str(x3)+" "+str(x4)+" \n")
        
    file.write("\n")
    file.write("POINT_DATA %i" % (nodos))
    file.write("\n")
    file.write("VECTORS Displacement float")
    file.write("\n")
    #Imprimiendo los valores de desplazamientos en orden
    for i in range(nodos):
        
        if i != (nodos-1):
            file.write(str(disp[i*2][0])+" "+str(disp[i*2+1][0])+ " "+ str(0)+" \n" )
            
        if i == (nodos-1):
            file.write(str(disp[i*2][0])+" "+str(disp[i*2+1][0])+ " "+ str(0) +" \n")

    file.close()

    return lista

import numpy as np
nombre = input("Ingrese nombre de la malla (sin formato):")
formato = ".txt"
nombre_malla = nombre+formato
Malla = open(nombre_malla,"r")
lineas = Malla.readlines()
Malla.close()
GAUSS = 6

displacement = displacement(GAUSS)
paraview_FEM(displacement)