
# Por si acaso se quiere matar el proceso de la base en activo
#sudo killall -9 pioneer

# Se da permisos de superusuario al puerto ttyUSB0
sudo chmod +x /dev/ttyUSB0

# Activamos rcnode
cd /home/robolab/robocomp/robocomp_tools/rcnode
./rcnode.sh &

# Entramos en la carpeta del componente           (Pon el path en tu caso)
cd /home/robolab/robocomp/components/robocomp-pioneer/components/pioneer_robot2

# Ejecutamos el componente
sudo bin/pioneer etc/config
