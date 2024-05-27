import psycopg2
from io import StringIO
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import settings as st
from config_setup import setup_service

def connect_database(serviceName):
    conf = setup_service(serviceName)
    hosts = conf.hosts
    databases = conf.databases
    users = conf.users
    passwords = conf.passwords
    ports = conf.ports
   
    # try:
    connect = psycopg2.connect(
        host=hosts,
        database=databases,
        user=users,
        password=passwords,
        port=ports
    )

    return connect

    # except (Exception, Error) as error:
    #     write_filelogging('error', 'Database connection Error : {}'.format(error))
    #     print("Error ",  error)
    #     return

def record_buffer(connect, cursor, table_tmName, data_df):
    buffer = StringIO()
    data_df.to_csv(buffer, index_label="id", header=False)
    buffer.seek(0)
    try:
        cursor.copy_from(buffer, table_tmName, sep=",")
        connect.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        # print("Error while The data is being recorded into {} table in database".format(table_tmName))
        # print("Error : {}".format(error))
        connect.rollback()
        return 1


def write_filelogging(level, messege):
    now = datetime.now()
    allfile = open(st.all_logpath, "a")
    if level == 'info':
        allfile.write("{} | INFO | {}\n".format(now, messege))
    elif level == 'warning':
        allfile.write("{} | WARNING | {}\n".format(now, messege))
        warnfile = open(st.warn_logpath, "a")
        warnfile.write("{} | WARNING | {}\n".format(now, messege))
        warnfile.close()
    elif level == 'error':
        allfile.write("{} | ERROR | {}\n".format(now, messege))
        errfile = open(st.error_logpath, "a")
        errfile.write("{} | ERROR | {}\n".format(now, messege))
        errfile.close()

    allfile.close()

def fetch_sql(sql, cur):
    try:
        cur.execute(sql)
        sqlfetch = cur.fetchall()
        return sqlfetch
    except Exception as err:
        write_filelogging('warning','Cannot query by \"{}\" | Error: {}'.format(sql, err))
        

def read_sql(sql,conn):
    try:
        sqlpd = pd.read_sql(sql,con=conn)
        return sqlpd  
    except: # if except, the function will reture None
        write_filelogging('warning','Cannot query by \"{}\"'.format(sql))


def close_database_server():
    print('-- close db ---')
    if st.tmanalysis_db_state == 1:
        st.connect_tmanalysis.close()
        st.cursor_tmanalysis.close()
        write_filelogging('info', 'Close connection {} database success | INFO'.format('tm_analysis'))
    if st.tmrecord_db_state == 1:
        st.connect_tmrecord.close()
        st.cursor_tmrecord.close()
        write_filelogging('info', 'Close connection {} database success | INFO'.format('tm_record'))
    if st.afs_mixer2_state == 1:
        st.connect_afs_mixer2.close()
        st.cursor_afs_mixer2.close()
        write_filelogging('info', 'Close connection {} database success | INFO'.format('mixer_dw_2021'))
    if st.mmgsserver1_state == 1:
        st.ftp.quit()
        write_filelogging('info', 'Close connection FTP MMGS server1')



def show_graphanomalydetection(real, reco, diff, anomaly_values, anomalystate_dataframe, resultanomaly_dataframe):
    color_line = ['#F9E79F', '#F5B041', '#e74c3c']

    fig = plt.figure(figsize=(15,10))
    # gs = GridSpec(nrows=4, ncols=2, width_ratios=[4,1])
    gs = GridSpec(nrows=4, ncols=1)
    ax0 = fig.add_subplot(gs[0,0])
    ax0.plot(real, label='real')
    ax0.plot(reco, label='reco')
    ax0.legend(loc='best')

    ax1 = fig.add_subplot(gs[1,0])
    ax1.plot(diff, label='score')
    for i in range(len(anomaly_values)):
        ax1.axhline(y=anomaly_values[i], label='anomaly level {}'.format(i+1), color=color_line[i])
    ax1.legend(loc='best')


    ax2 = fig.add_subplot(gs[2, 0])
    ax2.plot(real, label='real')
    for i in range(len(anomaly_values)):
        ax2.scatter(anomalystate_dataframe[anomalystate_dataframe['anomaly_state'] == i+1]['id'],
                    anomalystate_dataframe[anomalystate_dataframe['anomaly_state'] == i+1]['real'], 
                    label='anomaly level '+str(i+1), color=color_line[i])
    ax2.legend(loc='best')

    # ax3 = fig.add_subplot(gs[:, 1])
    # ax3.hist(diff, bins=100, label='score')
    # for i in range(len(anomaly_values)):
    #     ax3.axvline(
    #         x=anomaly_values[i], label='anomaly level '+str(i+1), color=color_line[i])
    # ax3.legend(loc='best')

    ax4 = fig.add_subplot(gs[3,0])
    ax4.plot(resultanomaly_dataframe['utc'], resultanomaly_dataframe['avg'])
    for i in range(len(anomaly_values)):
        ax4.scatter(resultanomaly_dataframe[resultanomaly_dataframe['anomaly_state_auto_m1'] == i+1]['utc'],
                    resultanomaly_dataframe[resultanomaly_dataframe['anomaly_state_auto_m1'] == i+1]['avg'],
                    label='anomaly level '+str(i+1), color=color_line[i])
    ax4.legend(loc='best')     

    fig.tight_layout(pad=.5, h_pad=2)
    plt.show()
