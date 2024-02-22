
# Lo primero es alcanzar la carpeta donde se encuentra
cd /home/robolab/robocomp/robocomp_tools/rcnode
./rcnode.sh &

# Despues se arranca el componente
cd /home/robolab/robocomp/components/robocomp-pioneer/components/gps_ublox

src/gps_ublox.py etc/config

