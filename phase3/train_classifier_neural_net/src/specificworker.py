from PySide2.QtCore import QTimer
from PySide2.QtWidgets import QApplication
from rich.console import Console
from genericworker import *
import interfaces as ifaces

sys.path.append('/opt/robocomp/lib')
console = Console(highlight=False)

# Importacion de librerias
import cv2 as cv            # Gestion de imágenes
import os                   # Gestion paths
import sys                  # Gestion exits
import numpy as np          # Gestion de matrices de imagenes
import random               # Valores random en la generacion del dataset

# Libreria pytorch necesarios para todo
import torch
import torch.nn as nn
from torch import optim



class SpecificWorker(GenericWorker):
    # Referencias de tiempo
    periodo = 30
    tiempoInicio = None

    # Referencias a la red neuronal y su entrenamiento
    redNeuronal = None
    optimizador = None

    # Rutas de directorios/ficheros/imagenes
    directorioObjetivo = "/media/robocomp/data_tfg/oficialDatasetPruebas/targetPerson"
    directorioNoObjetivo = "/media/robocomp/data_tfg/oficialDatasetPruebas/noTargetPerson"

    # Extra 
    device = torch.device("cpu")
    dataset = None
    contadorImagenes = 0

    # Constantes
    TAMANO_ENTRADA = [350, 150, 3]          # 350 (Altura), 150 (Anchura), 3 (Canales de color, RGB)
    MEZCLAR_DATOS_DATASET = False
    NUMERO_FRAMES_DATASET = 20
    TAMANO_BATCH = 32







    def __init__(self, proxy_map, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map)
        self.Period = self.periodo

        self.comprobar_condiciones ()

        self.construccion_red_neuronal_siamesa ()

        self.timer.timeout.connect(self.compute)
        self.timer.start(self.Period)

    def __del__(self):
        

        return

    def setParams(self, params):
        return True


    @QtCore.Slot()
    def compute(self):
        if self.contadorImagenes < self.NUMERO_FRAMES_DATASET:
            imagen1 = self.dataset[0][self.contadorImagenes]
            imagen2 = self.dataset[1][self.contadorImagenes]
            
            self.procesamiento_imagenes (imagen1, imagen2)

            

            if 


            self.contadorImagenes += 1

        else:
            sys.exit ("FIN: Se acabaron de procesar todos los frames")

        return True




    # ------------------------------------------------
    # ------------------ INITIALIZE ------------------
    # ------------------------------------------------

    def comprobar_condiciones (self):

        # Se comprueba si existe la ruta del dataset
        if not os.path.exists (self.directorioObjetivo) :            
            sys.exit (f"ERROR (1): No existe el directorio {self.directorioFrames}. Compruebelo bien.")
        
        if not os.path.exists (self.directorioNoObjetivo):
            sys.exit (f"ERROR (1): No existe el directorio {self.directorioFrames}. Compruebelo bien.")

        # Si se llega a este punto existen las dos carpetas
        self.cargar_dataset_del_disco ()

        return

    # ------------------------------------------

    # Construccion red neuronal
    def construccion_red_neuronal_siamesa(self):
        # Establecimiento de la secuencia de capas CNN
        cnn_layers = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )

        # Configuracion de capas FC
        fc_layers = nn.Sequential(
            nn.Linear(26112, 8192),
            nn.ReLU(inplace=True),
            nn.Linear(8192, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        # Combinando las capas CNN y FC
        self.redNeuronal = nn.Sequential(
            cnn_layers,
            nn.Flatten(),  
            fc_layers
        )

        self.optimizador = optim.Adam(self.redNeuronal.parameters(), lr = 0.0005 )

        return 

    # -------------------------------------------------
    # -------------------- COMPUTE --------------------
    # -------------------------------------------------
    
    def procesamiento_imagenes (self, image1, image2):


        return

    # ------------------------------------------------
    # -------------------- FINISH --------------------
    # ------------------------------------------------


    # -----------------------------------------------
    # -------------------- EXTRA --------------------
    # -----------------------------------------------


    # Separa los datos en 3 listas distintas (Mayor comodidad)
    def separacion_datos_dataset (self):
        ROIs1 = self.dataset[0]
        ROIs2 = self.dataset[1]
        similitud = self.dataset[2]

        return ROIs1, ROIs2, similitud
    
    def cargar_dataset_del_disco (self):

        # Se asume que ambas carpetas tienen contenido de imagenes (Solo imagenes)
        listaRutasAbsolutasImagenesObjetivo = [os.path.abspath(os.path.join(self.directorioObjetivo, file)) for file in os.listdir(self.directorioObjetivo)]
        listaRutasAbsolutasImagenesNoObjetivo = [os.path.abspath(os.path.join(self.directorioNoObjetivo, file)) for file in os.listdir(self.directorioNoObjetivo)]

        nuevoDataset = [[], [], []]

        for i in range (self.NUMERO_FRAMES_DATASET):
            # Se pone menos uno ya que si no puede dar lugar a index error
            posicionImagenAleatoria = random.randint (0, len (listaRutasAbsolutasImagenesObjetivo) - 1)

            # Se añade al dataset en la posicion 0 la imagen aleatoria redimensionada
            nuevoDataset[0].append (self.redimensionar_imagen (cv.imread (listaRutasAbsolutasImagenesObjetivo[posicionImagenAleatoria])))
            
            # Si es 0 se pondra de segundo frame uno de los no objetivos
            # Si es 1 se pondra de segundo frame uno del objetivo
            tipoSegundaImagen = random.randint (0, 1)

            # Se realiza con dos ifs en lugar de if else para que sea ampliable
            if tipoSegundaImagen == 0:
                posicionImagenAleatoria = random.randint (0, len (listaRutasAbsolutasImagenesNoObjetivo) - 1)
                nuevoDataset[1].append (self.redimensionar_imagen (cv.imread (listaRutasAbsolutasImagenesNoObjetivo[posicionImagenAleatoria])))
                nuevoDataset[2].append (0)

            if tipoSegundaImagen == 1:
                posicionImagenAleatoria = random.randint (0, len (listaRutasAbsolutasImagenesObjetivo) - 1)                   
                nuevoDataset[1].append (self.redimensionar_imagen (cv.imread (listaRutasAbsolutasImagenesObjetivo[posicionImagenAleatoria])))
                nuevoDataset[2].append (1)
        
        # Si se quiere mezclar el proceso cambia
        if self.MEZCLAR_DATOS_DATASET:
            a = 0

        # Si no es relativamente sencillo, solo se copia en la variable global
        else:
            self.dataset = nuevoDataset

        return

    # ----------------------------------------------

    def redimensionar_imagen (self, imagenOriginal):
        imagenRedimensionada = cv.resize (imagenOriginal, (self.TAMANO_ENTRADA [1], self.TAMANO_ENTRADA [0]))
        
        return imagenRedimensionada

    # ---------------------------------------

    def binary_cross_entropy(y_true, y_pred):
        """
        Binary Cross-Entropy Loss Function

        Parámetros:
        y_true: int
            Etiquetas verdaderas (verdad fundamental) en formato binario (0 o 1).
        y_pred: float
            Probabilidades predichas para cada muestra, que van de 0 a 1.

        Devoluciones:
        Float
            Pérdida de entropía cruzada binaria.
        """
        # Clip predicted values to avoid numerical instability (log(0) = -inf)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        # Calculate binary cross-entropy loss
        loss = - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

        return np.mean(loss)
    











    # ------------------------------------------------
    # ------------- INNECESARY / USEFULL -------------
    # ------------------------------------------------

    def print_dataset (self):

        print (f"longitud datset[0]: {len (self.dataset[0])}")
        print (f"longitud datset[1]: {len (self.dataset[1])}")
        print (f"longitud datset[3]: {len (self.dataset[2])}")

        for i in range (len (self.dataset[0])):
            cv.imshow ("imagen1", self.dataset[0][i])
            cv.imshow ("imagen2", self.dataset[1][i])
            print (f"imagen {i} -> {self.dataset[2][i]}")

            if cv.waitKey (0) == 27:
                sys.exit ("Fin")

        return
    