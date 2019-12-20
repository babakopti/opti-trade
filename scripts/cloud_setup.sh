# Do these before running this script
# sudo apt-get install git
# git clone https://github.com/babakopti/opti-trade.git

sudo apt-get install emacs
sudo apt-get install python3-pip
sudo pip3 install numpy pandas matplotlib scipy scikit-learn dill
sudo pip3 install schedule
sudo pip3 install requests

git config --global status.showUntrackedFiles no
git config --global user.email "optilive.developer@gmail.com"
git config --global user.name "Babak Emami"

sudo apt-get install build-essential checkinstall
sudo apt-get install libreadline-gplv2-dev libncursesw5-dev libssl-dev \
    libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libffi-dev zlib1g-dev


wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
sudo ./configure
sudo make

sudo make install
pip3 install ta-lib
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# If a newer pythopn3 version was required
#sudo wget https://www.python.org/ftp/python/3.8.0/Python-3.8.0.tgz
#sudo tar xzf Python-3.8.0.tgz
#cd Python-3.8.0
#sudo ./configure --enable-optimizations
#sudo make altinstall
