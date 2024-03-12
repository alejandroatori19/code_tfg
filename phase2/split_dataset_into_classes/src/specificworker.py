from PySide2.QtCore import QTimer
from PySide2.QtWidgets import QApplication
from rich.console import Console
from genericworker import *
import interfaces as ifaces

sys.path.append('/opt/robocomp/lib')
console = Console(highlight=False)


# Importacion de librerias
from ultralytics import YOLO
from PIL import Image
import cv2 as cv
import numpy as np
import time
import os
import json
import shutil


class SpecificWorker(GenericWorker):
    
    # Timers
    periodo = 15

    # Variables referentes a la red neuronal
    redNeuronalYolo = None 

    # Rutas y ficheros
    rutaDataset = "/media/robocomp/data_tfg/dataset/colorFrames"
    rutaDatasetClasificado = "/media/robocomp/data_tfg/sortedDataset"
    listaNombreFramesColor = []

    # Variables constantes
    tamanoFrame = (640, 480)
    tamanoComun = (500, 500)
    NUMERO_DECIMALES = 3

    # Flags
    CONCATENAR_IMAGENES = True
    SEPARAR_RESULTADOS = True
    FILTRAR_SOLO_PERSONAS = False
    GUARDAR_ROIS_EN_DISCO = True
    ACTUALIZAR_ROI = True
    PRECISION_MINIMA_ACEPTABLE = 0.7
    MSE_MAXIMO_ACEPTABLE = 10
    
    # Flags & variables modificables
    contadorFrames = None
    tiempoInicio = None
    tiempoMedioRedNeuronal = None    
    imagenOriginalTarget = None

    


    def __init__(self, proxy_map, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map)
        self.Period = self.periodo

        # Comprobacion de condiciones basicas
        self.comprobar_condiciones ()

        # Precarga de la red neuronal (Si no esta descargado se descargara automaticamente)
        self.redNeuronalYolo = YOLO ("yolov8s.pt")

        self.timer.timeout.connect(self.compute)
        self.timer.start(self.Period)

        return

    def __del__(self):

        return

    def setParams(self, params):


        return True


    @QtCore.Slot()
    def compute(self):
        # La condicion es que no se hayan trabajado todos los frames
        if self.contadorFrames < len (self.listaNombreFramesColor):
            
            # Ruta del frame que se va a leer y pasar por la red neuronal
            rutaFrame = self.rutaDataset + "/" + self.listaNombreFramesColor [self.contadorFrames]
            frame = cv.imread (rutaFrame)

            # Procesa la imagen en la red neuronal y devuelve unos resultados
            resultados = self.obtener_resultados_red_neuronal (frame)

            # Si el flag esta activo separa los resultados de la variable en listas (Para poder filtrar los datos de manera mas eficiente)
            if self.SEPARAR_RESULTADOS:            
                listaClases, listaPrecision, listaCajaColision = self.obtener_resultados_en_listas (resultados)
                
                # Filtra los datos obtenidos
                listaClases, listaPrecision, listaCajaColision = self.filtrar_resultados (listaClases, listaPrecision, listaCajaColision)

                if self.GUARDAR_ROIS_EN_DISCO:
                    self.obtencion_y_guardado_rois (frame, listaClases, listaCajaColision)


            # Lo ultimo sería mostrar al usuario como quedaría la imagen junto con la deteccion
            self.mostrar_frames_al_usuario (frame, resultados[0].plot ())

            # FInalmente se incrementa el valor del contador de frames ya que un frame ha sido procesado correctamente
            self.contadorFrames += 1
        
        else:
            sys.exit ("Se ha finalizado la ejecución correctamente. No hay mas frames para leer")
    
        #sys.exit ("TESTING")


        return True


    # ------------------------------------------------
    # ------------------ INITIALIZE ------------------
    # ------------------------------------------------
    
    def comprobar_condiciones (self):
        # Inicializa variables de gestion del código
        self.contadorFrames = 0
        self.tiempoInicio = time.time ()
        self.tiempoMedioRedNeuronal = 0
        self.imagenOriginalTarget = cv.imread ("/media/robocomp/data_tfg/dataset/originalTarget/original_1.jpeg")
    
        # Se comprueba si existe la ruta del dataset
        if os.path.exists (self.rutaDataset) :

            # Se obtienen los nombres de los frames que haya guardado en ambas carpetas. (Se esta suponiendo que el nombre es ese si no se modifica el otro codigo)
            self.listaNombreFramesColor = os.listdir (self.rutaDataset)

        else:
            sys.exit (f"ERROR (1): No existe el path {self.rutaDataset}. Compruebelo bien.")

        nombreCarpetaDataset = self.rutaDatasetClasificado [:self.rutaDatasetClasificado.rfind ('/')]

        if os.path.exists (nombreCarpetaDataset):
            # Si existe previamente se debe borrar
            if os.path.exists (self.rutaDatasetClasificado):
                shutil.rmtree(self.rutaDatasetClasificado)
                print ("AVISO: Se ha borrado la antigua carpeta \"sortedDataset\" para crearla de nuevo vacia.")
            
            # Se crea la carpeta pero vacia
            os.makedirs (self.rutaDatasetClasificado)

            # Dentro de esta carpeta previamente creada se añaden 2 mas para guardar el datasetOrdenado. Tanto las de persona objetivo como la del resto
            os.makedirs (self.rutaDatasetClasificado + "/targetPerson")
            os.makedirs (self.rutaDatasetClasificado + "/noTargetPerson")
            os.makedirs (self.rutaDatasetClasificado + "/otherTarget")

            print ("AVISO: El entorno del dataset ha sido generado correctamente")

        else:
            sys.exit (f"ERROR (2): No existe la ruta {nombreCarpetaDataset}. Compruebelo bien")


        return

    # -------------------------------------------------
    # -------------------- COMPUTE --------------------
    # -------------------------------------------------
    
    def obtener_resultados_red_neuronal (self, frame):
        # Se va a medir el tiempo que tardar en obtener resultados la red neuronal.
        tiempoInicioRedNeuronal = time.time ()

        # Primero se obtienen los resultados de la red neuronal.
        resultados = self.redNeuronalYolo (frame)

        # Calculo del tiempo medio e incremento a la variable global
        self.tiempoMedioRedNeuronal += (time.time () - tiempoInicioRedNeuronal)

        return resultados
    
    # -----------------------------------------------------------------------------

    def obtencion_y_guardado_rois (self, image, listaClases, listaCajaColision):
        # Se crea una lista que va a almacenar los rois
        listaRois = []
        numeroPersonas = sum(1 for clase in listaClases if clase == 0)
        contadorRois = 1

        print ("listaClases:", listaClases)
        print ("listaCajaColision:", listaCajaColision)

        # Lo primero que se hace es obtener los rois en una lista
        for cajaColisionDeteccion in listaCajaColision:
            regionInteres = image [cajaColisionDeteccion[1] : cajaColisionDeteccion[3],
                                   cajaColisionDeteccion[0] : cajaColisionDeteccion[2]]
            
            listaRois.append (regionInteres)
        
        indiceMayorSimilitud = self.indice_persona_mayor_similitud (listaClases, listaRois)

        # Después se van a guardar unicamente los que no hacen referencia a personas (Si solo hay personas omitimos este paso)
        for i in range (len (listaClases)):
            if listaClases[i] != 0:
                # Se obtiene la ruta donde se va a guardar
                rutaRegionInteres = self.rutaDatasetClasificado + "/otherTarget/image_" + str (self.contadorFrames) + "_" + str (contadorRois) + ".jpeg"

            else:
                if i == indiceMayorSimilitud:
                    rutaRegionInteres = self.rutaDatasetClasificado + "/targetPerson/image_" + str (self.contadorFrames) + "_" + str (contadorRois) + ".jpeg"

                else:
                    rutaRegionInteres = self.rutaDatasetClasificado + "/noTargetPerson/image_" + str (self.contadorFrames) + "_" + str (contadorRois) + ".jpeg"

            # Se guarda el roi en la carpeta correspondiente
            cv.imwrite (rutaRegionInteres, listaRois[i])

            # Se incrementa el contador para que no se llamen iguales
            contadorRois += 1


        return

    # -------------------------------------------------------------------

    def mostrar_frames_al_usuario (self, frameBase, frameConDetecciones):
        
        # Este flag indica que ambas imagenes se concatenen para mostrarse en una ventana o bien que se muestren cada una en la suya.
        if self.CONCATENAR_IMAGENES:
            if frameBase.shape != frameConDetecciones.shape:
                sys.exit ("ERROR (4): Estas intentando concatenar dos imagenes que no tienen los mismos tamaños. Compruebelo bien.")

            framesConcatenados = np.concatenate((frameBase, frameConDetecciones), axis=1) 

            cv.imshow ('Frames', framesConcatenados)

        else:
            cv.imshow ("Frame Base", frameBase)
            cv.imshow ("Frame Con Detecciones", frameConDetecciones)

        # Es un boton de seguridad si mientras la ejecucion presionas ESC entonces el programa se parara. Sirve mas para testing
        if cv.waitKey (0) == 27:
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
        print (f"Carpeta dataset: {self.rutaDataset}")
        print ("------------- FIN INFORMACION -------------\n\n")
        
        return

    # -----------------------------------------------
    # -------------------- EXTRA --------------------
    # -----------------------------------------------

    def obtener_resultados_en_listas (self, resultados):
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

        # Se devuelven los resultados en listas
        return listaClaseDeteccion, listaPrecisionDeteccion, listaCajaColisionDeteccion
    
    # ----------------------------------------------------------------------------

    def filtrar_resultados (self, listaClases, listaPrecision, listaCajaColision):
        # Se crean las listas vacias
        listaFiltradaClases = []
        listaFiltradaPrecision = []
        listaFiltradaCajaColisiones = []

        for i in range (len (listaPrecision)):
            # Se comprueba si cumple el minimo. Si no lo hace lo borra.
            if listaPrecision [i] > self.PRECISION_MINIMA_ACEPTABLE:
                listaFiltradaClases.append (listaClases[i])
                listaFiltradaPrecision.append (listaPrecision[i])
                listaFiltradaCajaColisiones.append (listaCajaColision [i])
            

        # Filtra solo por personas
        if self.FILTRAR_SOLO_PERSONAS:
            j = 0
            while j < len (listaFiltradaPrecision):
                if listaFiltradaClases [j] != 0:
                    del (listaFiltradaClases [j])
                    del (listaFiltradaPrecision [j])
                    del (listaFiltradaCajaColisiones [j])
                else:
                    j += 1

        return listaFiltradaClases, listaFiltradaPrecision, listaFiltradaCajaColisiones
    
    # -----------------------------------------------------------

    def indice_persona_mayor_similitud (self, listaClases, listaRois):
        # Variables generales
        indiceMayorSimilitud = 0
        valorMayorSimilitud = 500

        # Se prepara la imagen para ser comparada (En escala de grises y con tamaño indicado)
        if self.ACTUALIZAR_ROI or (self.contadorFrames == 0):
            imagenOriginal = cv.resize(self.imagenOriginalTarget, self.tamanoComun)
            imagenOriginal = cv.cvtColor(imagenOriginal, cv.COLOR_BGR2GRAY)

        # Se recorren los rois solo contando los de personas y se obtiene el que mas parecido tenga
        for i in range (len (listaClases)):
            if listaClases[i] == 0:
                # Obtencion del roi a comprobar ya preparado
                imagenComprobacion = listaRois[i]
                imagenComprobacion = cv.resize(imagenComprobacion, self.tamanoComun)
                imagenComprobacion = cv.cvtColor(imagenComprobacion, cv.COLOR_BGR2GRAY)

                # Calculo del grado similitud
                gradoSimilitud= ((imagenOriginal - imagenComprobacion) ** 2).mean()

                # Si supera al maximo hasta ahora se sobreescribe
                if gradoSimilitud < valorMayorSimilitud:
                    indiceMayorSimilitud = i
                    valorMayorSimilitud = gradoSimilitud

                print ("valorSimilitud:", gradoSimilitud)

                cv.imshow ('test', listaRois[i])
                cv.imshow ('imagen original:', self.imagenOriginalTarget)
                cv.waitKey (0)

        #if valorMayorSimilitud < self.MSE_MAXIMO_ACEPTABLE and self.ACTUALIZAR_ROI:
            #self.imagenOriginalTarget = listaRois[indiceMayorSimilitud]
            #print ("Actualizar Imagen")


        # Se hace para evitar que se equivoque en el target.
        if valorMayorSimilitud > self.MSE_MAXIMO_ACEPTABLE:
            indiceMayorSimilitud = -1

        return indiceMayorSimilitud




