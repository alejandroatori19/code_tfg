

from PySide2.QtCore import QTimer
from PySide2.QtWidgets import QApplication
from rich.console import Console
from genericworker import *
import interfaces as ifaces

sys.path.append('/opt/robocomp/lib')
console = Console(highlight=False)

# Import libraries
import pyrealsense2 as rs2
import numpy as np
import cv2 as cv
import os
import time
import shutil


class SpecificWorker(GenericWorker):

    # Timers
    periodo = 15

    # Variables referentes al video
    pipeline = None
    configuracion = None

    # Paths
    rutaVideo = "/media/robocomp/data_tfg/oficialVideos/video1.bag"
    rutaDataset = "/media/robocomp/data_tfg/dataset"
    
    # Variables constantes
    tamanoFrame = (640, 480)
    ROTAR_FRAMES = 0
    NUMERO_DECIMALES = 3

    # Flags
    GUARDAR_FRAMES = True    
    PIPELINE_ARRANCADA = False
    ACTIVAR_TODOS_LOS_STREAMS = True
    MOSTRAR_FRAMES_USUARIO = True
    COLORIZAR_FRAMES_PROFUNDIDAD = False
    CONCATENAR_FRAMES = False 
    
    # Flags & variables modificables
    contadorFrames = None
    tiempoInicio = None
    tiempoMedioPorFrame = None
    tiempoCargaVideo = None


    def __init__(self, proxy_map, startup_check=False):
        super(SpecificWorker, self).__init__(proxy_map)
        self.Period = self.periodo

        # Comprueba si ciertas condiciones necesarias se cumplen. También hace ciertas modificaciones para preparar el entorno
        self.comprobar_condiciones ()

        # Inicializa la conexion con el video guardado en un archivo.
        self.inicializar_video_desde_fichero ()

        # Activa el timer y lo conecta con el metodo compute
        self.timer.timeout.connect(self.compute)
        self.timer.start(self.Period)
        
        return

    def __del__(self):
        if self.contadorFrames > 0:
            self.mostrar_datos_al_usuario ()

        # Solo se para la pipeline si ha sido arrancada previamente
        if self.PIPELINE_ARRANCADA:
            self.pipeline.stop ()

        return

    def setParams(self, params):


        return True


    @QtCore.Slot()
    def compute(self):

        # Para calcular cuanto tiempo tarda en recoger el frame
        tiempoInicioFrame = time.time ()

        # Este metodo es el único que ofrece la posibilidad de comprobar si se ha recibido o no el frame. Es necesario para saber si hemos llegado al final
        frameDisponible, frame = self.pipeline.try_wait_for_frames ()
        
        # Se incrementa el tiempo medio por frame con lo que ha tardado el actual
        self.tiempoMedioPorFrame += (time.time () - tiempoInicioFrame)

        # Obviamente, solo se guarda en caso de que haya un frame disponible. Si no se asume que ya se ha acabado el video.
        if frameDisponible:
            # Modificacion del frame para que pueda ser guardado y mostrado al usuario.
            frameColorPreparado, frameProfundidadPreparado = self.preparacion_de_frame (frame)

            # Muestra al usuario las dos imagenes mediante opencv
            if self.MOSTRAR_FRAMES_USUARIO:
                self.mostrar_frames_al_usuario (frameColorPreparado, frameProfundidadPreparado)
        
            if self.GUARDAR_FRAMES:
                self.guardar_frames_en_disco (frameColorPreparado, frameProfundidadPreparado)

            # Incrementa el numero de frames que ya se han guardado en 1
            self.contadorFrames += 1
        else:
            sys.exit ("Fin programa. No se ha obtenido el frame")

        return True
    

    # ------------------------------------------------
    # ------------------ INITIALIZE ------------------
    # ------------------------------------------------
    
    def comprobar_condiciones (self):
        # Inicializa variables de gestion del código
        self.contadorFrames = 0
        self.tiempoInicio = time.time ()
        self.tiempoMedioPorFrame = 0

        # Si se quiere concatenar y no se esta colorizando el frame de profundidad entonces dará un error ya que no tendran el mismo tamaño
        # Y por tanto no se pueden concatenar. Esto lo unico que hace es preveerlo y hacer que el usuario no pierda el tiempo.
        if self.CONCATENAR_FRAMES and not self.COLORIZE_DEPTH_FRAME:
            sys.exit ("ERROR (1): Si se quiere concatenar los frames en la visualizacion del usuario, se necesita colorizar el de profundidad. Por favor, revise el valor del flag")
        
        # Este flag debe ser un valor numérico entre 0 y 3
        if self.ROTAR_FRAMES < 0 and self.ROTAR_FRAMES > 3:
            sys.exit ("ERROR (2): Comprueba si el valor del flag ROTAR_FRAMES se encuentra entre los valores validos 0 <= ROTAR_FRAMES <= 3")

        # Comprobacion si existe el fichero de video y que tenga la extension ".bag"
        if not (os.path.exists (self.rutaVideo) and self.rutaVideo[-3:] == "bag"):
            sys.exit ("ERROR (3): No existe el fichero de video o la ruta está mal escrita o la extension es incorrecta (Debe ser \".bag\"). Compruebalo y corrigela.")

        # Ahora se va a comprobar si existe la carpeta del dataset. En cuyo caso se borrara y creara de nuevo para sobreescribir lo que tuviese.
        nombreCarpetaDataset = self.rutaVideo [:self.rutaDataset.rfind ("/") + 1]

        print (self.rutaDataset)

        # Se comprueba si existe la carpeta main sin la carpeta donde se guarda el dataset
        if os.path.exists (nombreCarpetaDataset):
            
            # Si existe lo primero es comprobar si la carpeta dataset existe.
            if os.path.exists (self.rutaDataset):
                shutil.rmtree(self.rutaDataset)
                print ("AVISO: Se ha borrado la antigua carpeta dataset para crearla de nuevo vacia.")

            # Se crea la carpeta pero vacia
            os.makedirs (self.rutaDataset)

            # Dentro de esta carpeta previamente creada se añaden 2 mas para guardar el dataset. Tanto las de color como las de profundidad
            os.makedirs (self.rutaDataset + "/colorFrames")
            os.makedirs (self.rutaDataset + "/depthFrames")

            print ("AVISO: El entorno del dataset ha sido generado correctamente")

        #  Si no existe simplemente acaba el código.
        else:
            sys.exit ("ERROR (4): La carpeta para guardar el dataset no existe. Por favor, creala o cambia dicha ruta.")



        return

    # --------------------------

    def inicializar_video_desde_fichero (self):
        """
        Method to initialize the video pipeline.
            
        """
        tiempoInicioCargaVideo = time.time ()

        # Primero es crear un objeto de ambas clases sin contenido.
        self.pipeline = rs2.pipeline ()
        self.configuracion = rs2.config()
            
        # Se indica en la configuracion que queremos coger la informacion de un fichero y sin playback (Por eso se le asigna false)
        self.configuracion.enable_device_from_file (self.rutaVideo, False)
        
        # Se activan todos los streams disponibles o algunos concretos. No tiene sentido aquí pero para futuras modificaciones
        if self.ACTIVAR_TODOS_LOS_STREAMS:
            # Enable all streams the video had. Is the best way to avoid errors
            self.configuracion.enable_all_streams ()

        else:
            # Change here if you only want to activate a single stream
            self.configuracion.enable_stream (rs2.stream.color, self.tamanoFrame[0], self.tamanoFrame[1], rs2.format.bgr8, 30)
            self.configuracion.enable_stream (rs2.stream.depth, self.tamanoFrame[0], self.tamanoFrame[1], rs2.format.z16, 30)

        # Starts the pipeline
        self.pipeline.start (self.configuracion)
            
        self.PIPELINE_ARRANCADA = True

        self.tiempoCargaVideo = time.time () - tiempoInicioCargaVideo

        return

    # -------------------------------------------------
    # -------------------- COMPUTE --------------------
    # -------------------------------------------------
    
    def preparacion_de_frame (self, frame):
        # Lo primero es obtener en variables del metodo los dos frames.
        colorFrame = frame.get_color_frame().get_data()
        depthFrame = frame.get_depth_frame().get_data()

        # Posteriormente se convierte en un np.array que es lo necesario para rotarla en caso de necesitarla y para guardarla en disco (Al menos con opencv)
        colorFrameNumpy = np.asarray (colorFrame)
        depthFrameNumpy = np.asarray (depthFrame)

        # Se coloriza el frame de profundidad para poder hacerlo más facil de ver. (Solo interesa para testeo)
        if self.COLORIZAR_FRAMES_PROFUNDIDAD:
            depthFrameNumpy = cv.applyColorMap(cv.convertScaleAbs(depthFrameNumpy, alpha=0.03), cv.COLORMAP_JET)

        # Finalmente se rota en caso de ser necesario
        if self.ROTAR_FRAMES != 0:
            rotatedColorFrame = np.rot90 (colorFrameNumpy, self.ROTAR_FRAMES)
            rotatedDepthFrame = np.rot90 (depthFrameNumpy, self.ROTAR_FRAMES)

            return rotatedColorFrame, rotatedDepthFrame
        
        return colorFrameNumpy, depthFrameNumpy

    # ------------------------
    
    def mostrar_frames_al_usuario (self, frameColor, frameProfundidad):

        # Este flag indica que ambas imagenes se concatenen para mostrarse en una ventana o bien que se muestren cada una en la suya.
        if self.CONCATENAR_FRAMES:
            if frameColor.shape != frameProfundidad.shape:
                print ("Si tienes los flags COLORIZE_DEPTH_FRAME y CONCATENAR_FRAMES con valores distintos (False, True) probablemente es la causa del problema.")
                sys.exit ("ERROR (4): Estas intentando concatenar dos imagenes que no tienen los mismos tamaños. Compruebelo bien.")

            framesConcatenados = np.concatenate((frameColor, frameProfundidad), axis=1) 

            cv.imshow ('Frame RGBD', framesConcatenados)

        else:
            cv.imshow ("Frame RGB", frameColor)
            cv.imshow ("Frame Depth", frameProfundidad)

        # Es un boton de seguridad si mientras la ejecucion presionas ESC entonces el programa se parara. Sirve mas para testing
        if cv.waitKey (1) == 27:
            sys.exit ("FIN EJECUCION: Se presiono la tecla \"ESC\" para finalizar la ejecucion")

        return

    # ---------------------------------------------------------------

    def guardar_frames_en_disco (self, frameColor, frameProfundidad):
        # Lo primero es la ruta donde van a ir los dos frames (Solo se indica la carpeta porque dentro de ella se crean 2 mas y cada una guardará cada frame)
        rutaFrameColor = self.rutaDataset + "/colorFrames/image_" + str (self.contadorFrames + 1) + ".jpeg" 
        rutaFrameProfundidad = self.rutaDataset + "/depthFrames/image_" + str (self.contadorFrames + 1) + ".jpeg" 

        # Se guardan las imagenes en las rutas correspondientes
        cv.imwrite (rutaFrameColor, frameColor)
        cv.imwrite (rutaFrameProfundidad, frameProfundidad)


        return

    # ------------------------------------------------
    # -------------------- FINISH --------------------
    # ------------------------------------------------

    def mostrar_datos_al_usuario (self):
        # Calculo de datos concretos
        tiempoTranscurrido = time.time() - self.tiempoInicio
        self.tiempoMedioPorFrame /= (self.contadorFrames)

        # Imprimiendo datos por consola
        print ("\n\n--------------- INFORMACION ---------------")
        print (f"Tiempo transcurrido: {round (tiempoTranscurrido, self.NUMERO_DECIMALES)} segundos")        
        print (f"Tiempo medio por frame: {round (self.tiempoMedioPorFrame, self.NUMERO_DECIMALES)} segundos")
        print (f"Tiempo de carga del video: {round (self.tiempoCargaVideo, self.NUMERO_DECIMALES)} segundos")
        print (f"Imagenes procesadas: {self.contadorFrames}")
        print (f"Carpeta dataset: {self.rutaDataset}")
        print ("------------- FIN INFORMACION -------------\n\n")
        return
   