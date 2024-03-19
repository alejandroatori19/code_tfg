from PySide2.QtCore import QTimer
from PySide2.QtWidgets import QApplication
from rich.console import Console
from genericworker import *
import interfaces as ifaces

sys.path.append('/opt/robocomp/lib')
console = Console(highlight=False)

# Librerias necesarias
from ultralytics import YOLO
import numpy as np
import cv2 as cv
import json
import os
import time
import shutil

class SpecificWorker(GenericWorker):
    # Variables globales de clase

    # Tiempos
    periodo = 30
    tiempoInicio = None
    tiempoMedioRedNeuronal = None
    

    # Red neuronal
    redNeuronalYolo = None

    # Origen de los frames
    directorioFrames = "/media/robocomp/data_tfg/dataset/colorFrames"
    listaRutasAbsolutasFrames = []

    # Destino de frames
    directorioDatasetFiltrado = "/media/robocomp/data_tfg/filteredDataset"

    # Extra
    contadorFrames = None
    datosDeteccionesJson = {}
    imagenObjetivoAnterior = None

    # Constantes
    NUMERO_DECIMALES = 2
    FRAMES_PARA_AVISO = 50
    SOBREESCRIBIR_DIRECTORIO_DATASET = True
    MINIMA_PRECISION_ACEPTABLE = 0.80
    FILTRO_SOLO_PERSONAS = True                 # Practicamente debe estar en True
    MOSTRAR_DETECCIONES_USUARIO = True
    TAMANO_ROIS = (640, 480)
    LIMITE_MSE_ACEPTABLE = 50

    # Lista de clases en su id de clase
    listaClases = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                   "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                   "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                   "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
                   "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
                   "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
                   "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
                   "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
                   "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                   "teddy bear", "hair drier"]
    

    def __init__(self, proxy_map, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map)
        self.Period = self.periodo
        
        # Comprobacion de condiciones basicas
        self.comprobar_condiciones ()

        # Precarga de la red neuronal (Si no esta descargado se descargara automaticamente)
        self.redNeuronalYolo = YOLO ("yolov8s.pt")

        self.timer.timeout.connect(self.compute)
        self.timer.start(self.Period)

    def __del__(self):

        return

    def setParams(self, params):


        return True


    @QtCore.Slot()
    def compute(self):
        if self.contadorFrames < len (self.listaRutasAbsolutasFrames):
            # Se obtiene el frame
            frame = cv.imread (self.listaRutasAbsolutasFrames[self.contadorFrames])

            # Devuelve los resultados simplemente para obtener la imagen con detecciones. Ya se ha realizado todo el procesamiento y volcado de datos a fichero y directorios
            self.procesamientoFrame (frame)

            self.contadorFrames += 1
        else:
            # Se muestran los datos al usuario.
            self.mostrar_datos_al_usuario ()
            sys.exit ("FINAL: El código ha finalizado debido a que ya se han analizado todos los frames que había")
        return True


    # ------------------------------------------------
    # ------------------ INITIALIZE ------------------
    # ------------------------------------------------
    
    def comprobar_condiciones (self):
        # Inicializa variables de gestion del código
        self.contadorFrames = 0
        self.tiempoInicio = time.time ()
        self.tiempoMedioRedNeuronal = 0
        self.imagenObjetivoAnterior = cv.imread ("/media/robocomp/data_tfg/dataset/originalTarget/original_1.jpeg")
    
        # Se comprueba si existe la ruta del dataset
        if os.path.exists (self.directorioFrames) :            
            
            print (f"CORRECTO -> El directorio de frames {self.directorioFrames} existe.")

            # Se obtendran las rutas absolutas y se guardan en la lista anteriormente creada.
            files = os.listdir(self.directorioFrames)
            self.listaRutasAbsolutasFrames = [os.path.abspath(os.path.join(self.directorioFrames, file)) for file in files]
        else:
            sys.exit (f"ERROR (1): No existe el directorio {self.directorioFrames}. Compruebelo bien.")

        # Obtiene el nombre de la carpeta anterior a la del dataset. Para comprobar si existe
        nombreDirectorioDataset = self.directorioDatasetFiltrado [:self.directorioDatasetFiltrado.rfind ('/')]

        # Directorio de destino se comprueba si existe la carpeta
        if os.path.exists (nombreDirectorioDataset):
            # Lo primero que hay que hacer es ver si existen la carpeta que guarda el dataset
            if os.path.exists (self.directorioDatasetFiltrado):
                if self.SOBREESCRIBIR_DIRECTORIO_DATASET:
                    # Se borra la carpeta y se crea de nuevo (Se sobreescribe)
                    shutil.rmtree(self.directorioDatasetFiltrado)
                    os.makedirs (self.directorioDatasetFiltrado)
                    print (f"ACCION -> El directorio del dataset filtrado {self.directorioDatasetFiltrado} se ha sobreescrito.")
            else:
                # Si no existe,  se debe crear
                os.makedirs (self.directorioDatasetFiltrado)
                print (f"ACCION -> El directorio del dataset filtrado {self.directorioDatasetFiltrado} se ha creado.")


            # Si el flag esta activo en cualquier caso se deben crear las 3 carpetas
            if self.SOBREESCRIBIR_DIRECTORIO_DATASET:
                os.makedirs (self.directorioDatasetFiltrado + "/targetPerson")
                os.makedirs (self.directorioDatasetFiltrado + "/noTargetPerson")
                os.makedirs (self.directorioDatasetFiltrado + "/otherTarget")

                print (f"ACCION -> Dentro del directorio {self.directorioDatasetFiltrado} se han creado 3 carpetas (Clases del dataset).")


            # Si no se sobreescribe lo que hace es comprobar si las carpetas existen. Si no, las crea.
            else:
                if not os.path.exists (self.directorioDatasetFiltrado + "/targetPerson"):
                    os.makedirs (self.directorioDatasetFiltrado + "/targetPerson")

                if not os.path.exists (self.directorioDatasetFiltrado + "/noTargetPerson"):
                    os.makedirs (self.directorioDatasetFiltrado + "/noTargetPerson")

                if not os.path.exists (self.directorioDatasetFiltrado + "/otherTarget"):
                    os.makedirs (self.directorioDatasetFiltrado + "/otherTarget")

        else:
            sys.exit (f"ERROR (1): No existe el directorio {nombreDirectorioDataset}. Compruebelo bien.")

        return

    # -------------------------------------------------
    # -------------------- COMPUTE --------------------
    # -------------------------------------------------
    
    def procesamientoFrame (self, frame):
        # Lo primero es pasarlo por la red neuronal y medir el tiempo que tardar en obtener resultados la misma.
        tiempoInicioRedNeuronal = time.time ()

        # Primero se obtienen los resultados de la red neuronal.
        resultados = self.redNeuronalYolo (frame)

        # Calculo del tiempo medio e incremento a la variable global
        self.tiempoMedioRedNeuronal += (time.time () - tiempoInicioRedNeuronal)

        # Después se van a procesar los datos obtenidos separandolos en distintos arrays para simplificar su entendimiento
        (listaClases, listaPrecision, listaCajaColision) = self.obtencion_y_filtrado_de_resultados (resultados)

        # Este flag debe desactivarse para ir mas rapido en la generacion del dataset clasificado
        if self.MOSTRAR_DETECCIONES_USUARIO:
            self.mostrar_frames_al_usuario (frame, resultados[0].plot ())

        # Guardado de rois y de datos en archivo json
        self.procesamiento_datos_y_guardado_rois (frame, listaClases, listaPrecision, listaCajaColision) 


        return resultados

    # -------------------------------------------------------------------

    def mostrar_frames_al_usuario (self, frameBase, frameConDetecciones):
        
        # Este flag indica que ambas imagenes se concatenen para mostrarse en una ventana o bien que se muestren cada una en la suya.
        if frameBase.shape != frameConDetecciones.shape:
            sys.exit ("ERROR (4): Estas intentando concatenar dos imagenes que no tienen los mismos tamaños. Compruebelo bien.")

        framesConcatenados = np.concatenate((frameBase, frameConDetecciones), axis=1) 

        cv.imshow ('Frames', framesConcatenados)

        # Es un boton de seguridad si mientras la ejecucion presionas ESC entonces el programa se parara. Sirve mas para testing
        if cv.waitKey (1) == 27:
            sys.exit ("FIN EJECUCION: Se presiono la tecla \"ESC\" para finalizar la ejecucion")

        return

    # ------------------------------------------------
    # -------------------- FINISH --------------------
    # ------------------------------------------------

    def mostrar_datos_al_usuario (self):
        # Calculo de datos concretos
        tiempoTranscurrido = time.time() - self.tiempoInicio
        self.tiempoMedioRedNeuronal /= (self.contadorFrames)

        # Imprimiendo datos por consola
        print ("\n\n--------------- INFORMACION ---------------")
        print (f"Tiempo transcurrido: {round (tiempoTranscurrido, self.NUMERO_DECIMALES)} segundos")        
        print (f"Tiempo medio por frame: {round (self.tiempoMedioRedNeuronal, self.NUMERO_DECIMALES)} segundos")
        print (f"Imagenes procesadas: {self.contadorFrames}")
        print (f"Carpeta dataset: {self.directorioDatasetFiltrado}")
        print ("------------- FIN INFORMACION -------------\n\n")
        
        # Guardado del fichero JSON
        rutaFicheroJson = self.directorioDatasetFiltrado + "/data.json"
        with open (rutaFicheroJson, 'w') as ficheroJson:
            json.dump (self.datosDeteccionesJson, ficheroJson, indent=4)
        
        return

    # -----------------------------------------------
    # -------------------- EXTRA --------------------
    # -----------------------------------------------

    def obtencion_y_filtrado_de_resultados (self, resultados):
        # Creacion de listas vacias
        listaClaseDeteccion = [] 
        listaPrecisionDeteccion = []
        listaCajaColisionDeteccion = []

        # Se van a pasar los resultados a unas listas externas
        for deteccion in resultados[0].boxes:
            # Lo principal es siempre pasar el contenido de gpu u otra cosa a cpu.
            listaClaseDeteccion.append (int (deteccion.cls.to('cpu')[0]))
            listaPrecisionDeteccion.append (float (deteccion.conf.to('cpu')[0]) )
            
            # Se hace lo siguiente para que se convierta correctamente
            coordenadasDeteccion = []
            for coordenada in deteccion.xyxy.to('cpu')[0]:
                coordenadasDeteccion.append (int (coordenada.item()))

            listaCajaColisionDeteccion.append (coordenadasDeteccion)

        # Se van a filtrar las listas quitando las detecciones que no lleguen al minimo de accuraccy establecido y si esta activo el flag
        # Se borran tambien las que no sean personas (id persona -> 0)
        i = 0
        while i < len (listaClaseDeteccion):
            # Si no cumple el minimo de precision entonces se borra de la lista
            if listaPrecisionDeteccion[i] < self.MINIMA_PRECISION_ACEPTABLE or (self.FILTRO_SOLO_PERSONAS and listaClaseDeteccion[i] != 0):
                del (listaClaseDeteccion[i])
                del (listaPrecisionDeteccion[i])
                del (listaCajaColisionDeteccion[i])

            else:
                i += 1

        # Se devuelven los resultados en listas
        return listaClaseDeteccion, listaPrecisionDeteccion, listaCajaColisionDeteccion
    
    # ---------------------------------------------------------------------------------------
    
    # Si se filtra por personas. Solo se puede usar este metodo si el flag "FILTRO_SOLO_PERSONAS" esta en True
    def procesamiento_datos_y_guardado_rois (self, imagenOriginal, listaClaseDeteccion, listaPrecisionDeteccion, listaCajaColisionDeteccion):
        personaObjetivo = False
        contadorDeteccionesPorFrame = 0

        (indiceSimilitud, listaValoresMSE, listaROIs) = self.obtencion_rois_y_similitud (imagenOriginal, listaCajaColisionDeteccion)
        
        for i in range (len (listaClaseDeteccion)):
            claseDeteccion = listaClaseDeteccion[i]
            precisionDeteccion = listaPrecisionDeteccion[i]            
            cajaColisionDeteccion = listaCajaColisionDeteccion[i]

            roiDeteccion = listaROIs [i]

            print ("MSE -", listaValoresMSE[i])

            # Indica que se ha encontrado una similitud aceptable
            if indiceSimilitud != -1:
                # Es el objetivo
                if indiceSimilitud == i:
                    rutaDestinoRoi = self.directorioDatasetFiltrado + "/targetPerson/image_" + str (self.contadorFrames + 1) + "_" + str (contadorDeteccionesPorFrame + 1) + ".jpeg"
                    self.imagenObjetivoAnterior = roiDeteccion
                    personaObjetivo = True

                # No es el objetivo
                else:
                    rutaDestinoRoi = self.directorioDatasetFiltrado + "/noTargetPerson/image_" + str (self.contadorFrames + 1) + "_" + str (contadorDeteccionesPorFrame + 1) + ".jpeg"

            # Si no se encuentra una simiilitud aceptable
            else:
                # Se muestra hasta que se pulse una tecla valida
                # ESC -> Acaba el programa
                # ENTER -> Es el objetivo
                # SPACE -> No es el objetivo
                while True:

                    # Si hay mas de una persona en el array lo que nos interesa es encontrar a la persona objetivo. La decide el usuario
                    cv.imshow ("Es la persona objetivo", roiDeteccion)
                    cv.imshow ("Anterior Objetivo", self.imagenObjetivoAnterior)
                    # Se hace bloqueante de manera que el código no siga hasta que la decision no se tome. Ira tomando decisiones en funcion de si es la persona objetivo o no
                    letraPulsada = cv.waitKey (0)
                        
                    # Si pulsa ENTER -> es la persona objetivo
                    if letraPulsada == 13:
                        print ("Si es el objetivo")

                        rutaDestinoRoi = self.directorioDatasetFiltrado + "/targetPerson/image_" + str (self.contadorFrames + 1) + "_" + str (contadorDeteccionesPorFrame + 1) + ".jpeg"

                        self.imagenObjetivoAnterior = roiDeteccion

                        # Se ha encontrado a la persona objetivo lo cual significa que el resto van a ser no objetivo
                        personaObjetivo = True

                        # Se aprovecha el bug propio del codigo de que solo considera como -1 el que no existe objetivo claro.
                        # Se le va a asignar el indice -2 ya que nunca será alcanzado y todos serán no objetivos (Ya se encontro manualmente al objetivo real)
                        indiceSimilitud = -2

                        break

                    # Si pulsa SPACE -> no es la persona objetivo
                    elif letraPulsada == 32:
                        print ("No es el objetivo")

                        rutaDestinoRoi = self.directorioDatasetFiltrado + "/noTargetPerson/image_" + str (self.contadorFrames + 1) + "_" + str (contadorDeteccionesPorFrame + 1) + ".jpeg"
                        
                        break

                    elif letraPulsada == 27:
                        self.mostrar_datos_al_usuario ()
                        sys.exit ("FIN: Se  ha pulsado la tecla ESC para finalizar la ejecucion.")
                            

                    else:
                        print ("ERROR: No se ha pulsado una tecla valida. ENTER -> Persona objetivo, SPACE -> Persona no objetivo")

            # Se guarda la imagen en disco
            cv.imwrite (rutaDestinoRoi, roiDeteccion)

            self.actualizacion_datos_fichero_json (claseDeteccion, precisionDeteccion, 
                                                   cajaColisionDeteccion, personaObjetivo,
                                                   contadorDeteccionesPorFrame + 1, rutaDestinoRoi, listaValoresMSE[i])

            # Reset e incremento de variables
            contadorDeteccionesPorFrame += 1
            personaObjetivo = False
        return
    
    # --------------------------------------

    def actualizacion_datos_fichero_json (self, clase, precision, cajaColision, esObjetivo, contadorDeteccionesPorFrame, rutaRoi, similitudObjetivoAnterior):
        
        nombreImagenOrigen = os.path.basename (self.listaRutasAbsolutasFrames[self.contadorFrames])
        nombreImagenOrigen = nombreImagenOrigen[:nombreImagenOrigen.find ('.')]

        # Se crea un diccionario de las cajas de colision
        cajaColision = {"x1" : str (cajaColision[0]),
                        "y1" : str (cajaColision[1]),
                        "x2" : str (cajaColision[2]),
                        "y2" : str (cajaColision[3])}
        
        # Genera un diccionario con el dato completo de la deteccion
        nuevoDatoDeteccion = {"id_detection" : str (contadorDeteccionesPorFrame),
                              "path_original_image" : self.listaRutasAbsolutasFrames[self.contadorFrames],
                              "path_roi" : rutaRoi,
                              "label_detection" : self.listaClases[clase],
                              "confidence:" : str (precision),
                              "is_target" : str(esObjetivo),
                              "mse_previous_target" : similitudObjetivoAnterior,
                              "boundingBox" : cajaColision }

        # Si no hay una clave referente al nombre de la imagen entonces se crea una y su valor sera una lista
        if nombreImagenOrigen not in self.datosDeteccionesJson.keys():
            self.datosDeteccionesJson[nombreImagenOrigen] = [] 
        
        # Se añaden al diccionario original
        self.datosDeteccionesJson[nombreImagenOrigen].append (nuevoDatoDeteccion)

        return
    
    # -------------------------------------------------------------------------

    def obtencion_rois_y_similitud (self, imagenOriginal, listaCajaColisiones):
        listaValoresMSE = []
        listaROIs = []
        encontradaSimilitud = -1

        indiceContador = 0    
        mseConSimilitud = 500

        # Recorre la lista de caja de colisiones para calcular la similitud con la anterior del objetivo
        for cajaColision in listaCajaColisiones:
            imagenROI = imagenOriginal [int (cajaColision[1]) : int (cajaColision[3]), 
                                        int (cajaColision[0]) : int (cajaColision[2])]
            
            # Redimensionalizacion de imagenes
            imagenObjetivoRedimensionada = cv.resize(self.imagenObjetivoAnterior, (self.TAMANO_ROIS[1], self.TAMANO_ROIS[0]))
            imagenRedimensionadaROI = cv.resize(imagenROI, (self.TAMANO_ROIS[1], self.TAMANO_ROIS[0]))

            # Conversion a escala de grises
            imagenObjetivoGris = cv.cvtColor(imagenObjetivoRedimensionada, cv.COLOR_BGR2GRAY)
            imagenGrisROI = cv.cvtColor(imagenRedimensionadaROI, cv.COLOR_BGR2GRAY)

            # Se calcula el nivel de similitud
            mse = ((imagenObjetivoGris - imagenGrisROI) ** 2).mean()

            # Se añade a los vectores
            listaROIs.append (imagenROI)
            listaValoresMSE.append (mse)

            # Se actualiza el valor si mejora el anterior
            if mse < mseConSimilitud and mse < self.LIMITE_MSE_ACEPTABLE:
                encontradaSimilitud = indiceContador

            # Aumenta el valor del contador
            indiceContador += 1


        return encontradaSimilitud, listaValoresMSE, listaROIs
