sudo ufw disable
ifconfig enp2s0f1   ### check whichever network adapter is running
sudo avahi-autoipd enp2s0f1
#avahi-browse -a -r   #just for verbose, should show robot serial