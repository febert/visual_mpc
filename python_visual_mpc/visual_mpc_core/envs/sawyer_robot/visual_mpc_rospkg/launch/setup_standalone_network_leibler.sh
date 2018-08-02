sudo ufw disable
ifconfig eth0
sudo avahi-autoipd eth0
#avahi-browse -a -r   #just for verbose, should show robot serial