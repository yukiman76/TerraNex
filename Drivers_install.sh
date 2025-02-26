https://www.civo.com/docs/compute/gpu-instance-drivers-ubuntu
sudo apt update && sudo apt upgrade -y
sudo apt autoremove nvidia* --purge
sudo reboot now
sudo apt install ubuntu-drivers-common -y
# tried 
# 535, 570 and now 
sudo apt install  nvidia-driver-550-open
sudo apt install nvidia-cuda-toolkit -y
sudo reboot now
nvidia-smi
nvcc --version

