pc_validation=${1:-0.03}
samples=${2:-10000}

git clone https://github.com/andrewmlu/data-sharing-tuor.git
git pull

# need python<=3.7 https://stackoverflow.com/questions/63168301/how-to-change-the-python-version-from-default-3-5-to-3-8-of-google-colab

sudo apt-get update -y
sudo apt-get install python3.7
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1

# Choose one of the given alternatives:
sudo update-alternatives --config python3

# This one used to work but now NOT(for me)!
# !sudo update-alternatives --config python

# Check the result
python3 --version

# Attention: Install pip (... needed!)
sudo apt install python3-pip

sudo add-apt-repository ppa:deadsnakes/ppa  # debug -- ModuleNotFoundError: No module named 'distutils.util' https://askubuntu.com/questions/1239829/modulenotfounderror-no-module-named-distutils-util
sudo apt-get update
sudo apt install python3.7-distutils

python3 -m pip install --upgrade pip setuptools  # debug -- ERROR: No matching distribution found for tensorflow==1.15.2 https://github.com/tensorflow/tensorflow/issues/34302

python --version

pip install -r requirements.txt

mkdir datasets/dataset_files/SVNH
mkdir datasets/dataset_files/cifar-gray-28by28
mkdir datasets/dataset_files/svnh-gray-28by28

cd datasets
python3 extract_svhn.py

cd ..
python3 main.py --config config_target_fashion --approach 0  --samples_per_dataset $samples --pc_validation $pc_validation --sim 1