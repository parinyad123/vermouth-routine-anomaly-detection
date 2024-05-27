import configparser as cf
from types import SimpleNamespace

import settings as st

def setup_service(serviceName):

    # path_dbconfig = os.path.join(os.path.dirname(__file__), "configDB.ini")
    path_dbconfig = ("".join([st.root_path,"/configDB.ini"]))
    # print("Path = ", path_dbconfig)
    setup = cf.ConfigParser()
    setup.read(path_dbconfig)

    if serviceName.split('_')[-1] == 'db':
        setup_config = {
            'hosts': setup.get(serviceName, 'hosts'),
            'databases' : setup.get(serviceName,'databases'),
            'users' : setup.get(serviceName,'users'),
            'passwords' : setup.get(serviceName,'passwords'),
            'ports' : setup.getint(serviceName, 'ports') ,
        }
    elif serviceName.split('_')[-1] == 'server':
        setup_config = {
            'hosts': setup.get(serviceName, 'hosts'),
            'users' : setup.get(serviceName,'users'),
            'passwords' : setup.get(serviceName,'passwords'),
            'ports' : setup.getint(serviceName, 'ports') ,
        }
    elif serviceName.split('_')[-1] == 'date':
        setup_config = {
            'date_start': setup.get(serviceName, 'date_start'),
            'date_end': setup.get(serviceName, 'date_end')
        } 
  
    setup_config = SimpleNamespace(**setup_config)

    return setup_config


def config_log():
    pass

if __name__ == "__main__":
    serviceName = 'MIXERs2021_DW_db'
    conf = setup_service(serviceName)
    print(conf)
    print(conf.hosts)
