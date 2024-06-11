1. install docker
$ sudo apt-get remove docker docker-engine docker.io
$ sudo apt-get update
$ sudo apt install docker.io
$ sudo snap install docker
$ docker --version


https://www.youtube.com/watch?v=LLqUFxAPsvs
https://www.youtube.com/watch?v=0e_C1B8fDvg

1. Download golang 1.16
$ wget https://dl.google.com/go/go1.16.linux-amd64.tar.gz
2. Extract tar.gz Files 
$  tar -xvzf go1.16.linux-amd64.tar.gz
3. Move go to usr/local
$ sudo mv go /usr/local/
4. Add golang path
$ nano ~/.bashrc
-> export PATH=$PATH:/usr/local/go/bin

Test backend (https://www.baeldung.com/curl-rest)
$ curl -v http://localhost:3020/



https://gist.github.com/heapwolf/3144998
1. Update APT index
$ sudo apt update
2. Install node 14.x
$ curl -sL https://deb.nodesource.com/setup_14.x | sudo bash -
$ sudo apt -y install nodejs
3. Check node version
$ node -v
4. Install npm
$ npm install


rundaily
1. Install pip for Python 3
$ sudo apt update
$ sudo apt install python3-pip

Create Virtual Environment
$ sudo apt install virtualenv
$ virtualenv --python=python3 env

Activate Virtual Environment
$ source env/bin/activate


2. Install 
$ pip install -r requirements.txt

3. Install psycopy2
$ sudo apt-get update
$ sudo apt-get -y install python3-pip
$ sudo apt install libpq-dev python3-dev
$ pip3 install psycopg2

Install other library
- pytorch 1.11.0
$ pip install torch==1.11.0+cpu torchvision==0.12.0+cpu torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cpu
- scipy 1.7.3
$ pip install scipy==1.7.3
- pandas 1.3.4
$ pip install pandas==1.3.4
- matplotlib 3.5.1
$ pip install matplotlib==3.5.1
- rootpath 0.1.1
$ pip install rootpath==0.1.1



change rootpath variable in settings.py file to run in server
- for Windows: root_path = rootpath.detect() 
- for Ubuntu : root_path = os.path.dirname(os.path.realpath(__file__))
change log directory to log_dir directory in settings.py
- for Windows: log_path = ("".join([root_path, "/log"]))
- for Ubuntu : log_path = ("".join([root_path, "/log_dir"]))
