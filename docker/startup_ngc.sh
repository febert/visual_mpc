curl -fsSL get.docker.com -o get-docker.sh;
sudo sh get-docker.sh;
sudo usermod -aG docker $USER;
cd ~/;
git clone https://github.com/febert/visual_mpc.git;
cd visual_mpc;
git checkout integrate_env;
git pull origin integrate_env;