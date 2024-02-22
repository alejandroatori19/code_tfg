
# Just kill all the process refered to piooner
sudo killall -9 pioneer

# That´s for the rcnode & command
sudo fuser -k 9999/tcp
sudo fuser -k 9998/tcp
