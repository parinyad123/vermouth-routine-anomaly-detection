
Routine Anomaly Detection
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
