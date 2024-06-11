import global_function as gf
import deleterow as delrow
import settings as st

from datetime import datetime, timezone
import os
import pandas as pd
import numpy as np
import torch
from torch.autograd import Variable
from torch import nn
from ftplib import FTP
import sys
import time
from datetime import datetime, timedelta

def prepare_preprocessing():
    print("LOG_PATH = ", st.log_path)
    #  === check log directory path
    if os.path.exists(st.log_path) == False:
        os.mkdir(st.log_path, 0o666)

    logfilelist = [st.all_logpath, st.warn_logpath, st.error_logpath]
    for logfile in logfilelist:
        # if .log file is not exists. it will be create
        if os.path.exists(logfile) == False:
            # # print(' log -> {}'.format(logfile))
            f = open(logfile, 'a')
            f.close()
            gf.write_filelogging('info', 'Create {}'.format(logfile))

    #  === check element .py file
    scriptloss_state = 0
    scriptloss = ''
    # checkfile_list = ['config_setup.py', 'configDB.ini',
    #                   'global_function.py', 'settings.py', 'models/model_auto_m1.py']
    checkfile_list = [st.config_setup_file, st.configDB_file, st.global_function_file, st.settings_file, st.model_file]
    for progfile in checkfile_list:
        print("Check file= ", progfile)
        # if not os.path.exists(("".join([st.root_path, "/{}".format(progfile)]))):
        if not os.path.exists(progfile):
            scriptloss = ("".join([scriptloss, " {},".format(progfile)]))
            scriptloss_state = 1
    if scriptloss_state == 1:
        gf.write_filelogging(
            'warning', 'Script files are loss : {}'.format(scriptloss))
        # gf.create_logging()
        return

    #  === check existing of archive model directory and empty directory (without model (.pt) files)
    if not os.path.exists(st.model_archivelocal_path):
        gf.write_filelogging('warning', 'The archive model directory does not exist')
        return
    else:
        model_items = os.listdir(st.model_archivelocal_path)
        if len(model_items) == 0:
            gf.write_filelogging('warning', 'The archive model directory is empty, containing no files(.pt)')
            return

    #  === connect DB
    try:
        st.db_ana = 'DEV_VERMOUTH_tm_analysis_db'
        st.db_rec = 'DEV_VERMOUTH_tm_record_db'
        st.db_mix2 = 'MIXERs2021_DW_db'

        st.messagelog = st.db_ana
        st.connect_tmanalysis = gf.connect_database(st.db_ana)
        st.cursor_tmanalysis = st.connect_tmanalysis.cursor()
        # gf.write_logging('info', 'Connect {} success'.format(st.db_ana))
        st.tmanalysis_db_state = 1
        gf.write_filelogging('info', 'Connect {} database success | INFO'.format('tm_analysis'))

        st.messagelog = st.db_rec
        st.connect_tmrecord = gf.connect_database(st.db_rec)
        st.cursor_tmrecord = st.connect_tmrecord.cursor()
        # gf.write_logging('info', 'Connect {} success'.format(st.db_rec))
        st.tmrecord_db_state = 1
        gf.write_filelogging('info', 'Connect {} database success | INFO'.format('tm_record'))

        st.messagelog = st.db_mix2
        st.connect_afs_mixer2 = gf.connect_database(st.db_mix2)
        st.cursor_afs_mixer2 = st.connect_afs_mixer2.cursor()
        # gf.write_logging('info', 'Connect {} success'.format(st.db_rec))
        st.afs_mixer2_state = 1
        gf.write_filelogging('info', 'Connect {} database success | INFO'.format('mixer_dw_2021'))

    except (Exception) as error:
        gf.write_filelogging('error', 'Failed to connect to {} database | Error: {}'.format(st.messagelog, error))
        gf.close_database_server()
        return

    #  === import list of TMs which have process_id == 3
    th1_tmprogress_sql = """SELECT tmname FROM th1_tmprogress WHERE progress_id = 3;"""
    prog = gf.read_sql(th1_tmprogress_sql, st.connect_tmrecord)
    # if prog=None, all program is error
    if prog.empty == True:
        gf.write_filelogging('warning', 'Cannot retrieve TM parameters which have model from th1_tmprogress table | Cannot retrieve data')
        return
    elif prog.empty == False:
        st.active_tm_record = [prog['tmname'][i] for i in range(len(prog))]
        st.prepare_tmrecord_state = True
    # list of TMs which have process_id == 3
    print("---->", st.prepare_tmrecord_state)

    # test 4 TM
    # st.active_tm_record = st.active_tm_record[120:130]

    # remove duplicated tm
    st.active_tm_record = list(set(st.active_tm_record))
    print(" ==> TM = ", st.active_tm_record)

def prepare_tmrecord():
    print("  Start prepare tm record")
    for tmname in st.active_tm_record:
        print("prepare_tmrecord = ", tmname)
        try:
            # query data for each tm from mixer2 between latest date of tm_record to current date
            data_dw = pd.DataFrame(
                columns=['name', 'generation_time', 'eng_value'])
            record_table = ("".join(['record_theos_', tmname])).lower()
            last_idtime_sql = """SELECT MAX(id) AS id, CAST(MAX(generation_time) AS DECIMAL(38,0)) AS gentime, max(utc) FROM {};""".format(record_table)

            #last_id = st.cursor_tmrecord.fetchall()
            last_idtime = gf.fetch_sql(last_idtime_sql, st.cursor_tmrecord)
         
            if last_idtime == []:
                gf.write_filelogging('warning', 'Cannot retrieve the data in {} table in prepare tmrecord process | Cannot retrieve data'.format(record_table))
            else:
                last_id = last_idtime[0][0]
                last_epoch_inc = last_idtime[0][1]+1
                epoch_now = int(datetime.now().timestamp()*1e9)
                update_tm_mix2_sql = """SELECT name, generation_time, eng_value FROM tm_param_afs WHERE name = \'{}\'
                                AND generation_time BETWEEN {} AND {};""".format(tmname, last_epoch_inc, epoch_now)
                raw_dw = gf.read_sql(update_tm_mix2_sql, st.connect_afs_mixer2)
                # if tm data from mixer2 not update -> cancel the tm
                # if tm data from mixer2 is update -> process continue
                print("Query Mixer2 = ",update_tm_mix2_sql)

                if raw_dw.empty == True:
                    gf.write_filelogging('warning', 'The data of {} was not updated in MIXER-II | Not found undated data'.format(tmname))
                else:
                    data_dw = data_dw.append(raw_dw, ignore_index=True)

                    # Clean tm raw data
                    data_dw['eng_value'] = pd.to_numeric(
                        data_dw['eng_value'], errors='coerce')
                    data_dw = data_dw.dropna()
                    if data_dw.empty == True:
                        gf.write_filelogging('warning', 'The updated data of {} which was updated in MIXER-II is NaN value | The data is Nan'.format(tmname))
                        return
                    else:
                        try:
                            # Clean tm raw data (continue)
                            data_dw = data_dw.drop_duplicates()
                            data_dw = data_dw.sort_values(by=['generation_time'])
                            data_dw.reset_index(inplace=True)
                            data_dw = data_dw.drop(['index'], axis=1)

                            # Convert generation_time to numeric type
                            data_dw['generation_time'] = pd.to_numeric(data_dw['generation_time'], errors='coerce')
                            data_dw['eng_value'] = pd.to_numeric(data_dw['eng_value'], errors='coerce')
                            print(data_dw)
                            # create UTC in data_dw
                            def convert_generationtimetoutc(epoch):
                                epoch = int(epoch/1e9)
                                date_str = datetime.utcfromtimestamp(epoch).strftime('%Y-%m-%d %H:%M:%S')
                                return datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
                            data_dw['utc'] = data_dw.apply(
                                lambda x: convert_generationtimetoutc(x['generation_time']), axis=1
                            )
                            data_dw.index = data_dw.index + last_id + 1
                        except Exception as err:
                            gf.write_filelogging(
                                'warning', 
                                'Data cleaning and preparing of {} is error in prepare tmrecord process | Error: {}'.
                                format(tmname, err))

                        st.active_tm.append(tmname)
                        
                        # record to tm_record table
                        gf.record_buffer(st.connect_tmrecord, st.cursor_tmrecord, record_table, data_dw)
                        st.createfeature_state = True
        except:
            continue
    
        # End Preparation tm record function

    # === Create Feature


def create_feature(tmname, freq, featuretable):
    print(' --- start Create Feature ----')
    # retrive parameters from analysis_params_theos_auto_m1 for clean raw data
    # param_tm_sql = """SELECT feature_table, lower_bound, upper_bound, delete_0value FROM analysis_params_theos_auto_m1 WHERE tm_name = \'{}\' and freq =  \'{}\';""".format(tmname, freq)
    param_tm_sql = """SELECT feature_table, lower_bound, upper_bound, delete_0value FROM analysis_params_theos_auto_m1 WHERE feature_table =  \'{}\';""".format(featuretable)
    param_tm = gf.read_sql(param_tm_sql, conn=st.connect_tmanalysis)
    print('= ana param_tm_sql = ', param_tm_sql)
    print('= ana param_tm => ', param_tm)
    # param_tm = pd.DataFrame()
    if param_tm.empty == True:
        gf.write_filelogging(
            'warning', 
            'Cannot retrive the data of {} with {} parameter in analysis_params_theos_auto_m1 table for create feature process | Cannot retrive data'.
            format(tmname, freq))
        return
    else:
        feature_table = param_tm['feature_table'][0]
        lower_bound = param_tm['lower_bound'][0]
        upper_bound = param_tm['upper_bound'][0]
        delete_zero = param_tm['delete_0value'][0]

    # retrive latest id, utc from analysis_theos_{TM}_{}
    last_idtime_sql = """SELECT max(id) as id, max(utc) as utc FROM {};""".format(feature_table)
    # print(last_idtime_sql)
    last_idtime = gf.read_sql(last_idtime_sql, conn=st.connect_tmanalysis)
    if last_idtime.empty == True:
        gf.write_filelogging(
        'warning', 
        'Cannot retrive the latest id and utc of {} table for create feature process | Cannot retrive data'.
        format(feature_table))
    else:
        # print(last_idtime)
        last_id_ana = last_idtime['id'][0]
        last_utc_ana = last_idtime['utc'][0]

        # retrive updated data from record_{TM} table
        record_tablename = ("".join(['record_theos_', tmname])).lower()
        record_data_sql = '''SELECT utc, eng_value FROM {} WHERE utc > \'{}\';'''.format(
            record_tablename, last_utc_ana)
        # print(record_data_sql)
        record_data = gf.read_sql(record_data_sql, conn=st.connect_tmrecord)
        # print(record_data)

    # clean data
    if record_data.empty == True:
        gf.write_filelogging(
            'warning', 
            'Cannot retrive utc and eng_value of {} table for create feature process | Cannot retrive data'.
            format(record_tablename))
    else:
        try:
            print('-- clean --')
            # delete zero value
            if delete_zero == True:
                index_name = record_data[record_data['eng_value'] == 0].index
                record_data = record_data.drop(index_name)
                # print('--- delete 0 value ---')
                # print(record_data)
            # delete boundary
            if record_data.empty == False:
                index_name = record_data[record_data['eng_value']
                                         < lower_bound].index
                record_data = record_data.drop(index_name)
                index_name = record_data[record_data['eng_value']
                                         > upper_bound].index
                record_data = record_data.drop(index_name)
            # reset index
            if record_data.empty == False:
                record_data.reset_index(inplace=True)
                record_data = record_data.drop(['index'], axis=1)
            clean_data = record_data.copy()
            # Clear memory of record_data = None
            record_data = None
        except Exception as err:
            gf.write_filelogging(
                'warning', 
                'Data cleaning of {} with {} is error in create feature process | Error: {}'.
                format(tmname, freq, err))
            return

    # Calculate to create feature
    if clean_data.empty == True:
        gf.write_filelogging(
            'warning', 
            'The data of {} with {} is loss after data was cleaned in create feature process | The data is loss'.
            format(tmname, freq))
    else:
        # print('-- calculate --')
        # print(clean_data)
        # print()
        # print('freq = ', freq)
        try:
            freqs = freq
            data_feature = clean_data.groupby(pd.Grouper(key='utc', freq=freqs)).mean().rename(columns={'eng_value': 'avg'})
            data_feature['std'] = clean_data.groupby(pd.Grouper(key='utc', freq=freqs)).std()
            data_feature['count'] = clean_data.groupby(pd.Grouper(key='utc', freq=freq)).count()
            data_feature['min'] = clean_data.groupby(pd.Grouper(key='utc', freq=freq)).min()
            data_feature['max'] = clean_data.groupby(pd.Grouper(key='utc', freq=freq)).max()
            data_feature['q1'] = clean_data.groupby(pd.Grouper(key='utc', freq=freq)).quantile(.25)
            data_feature['q2'] = clean_data.groupby(pd.Grouper(key='utc', freq=freq)).quantile(.5)
            data_feature['q3'] = clean_data.groupby(pd.Grouper(key='utc', freq=freq)).quantile(.75)
            data_feature['skew'] = clean_data.groupby(pd.Grouper(key='utc', freq=freq)).skew()
            data_feature['lost_state'] = pd.isna(data_feature['avg'])

            orders = 1
            data_feature['avg'] = data_feature['avg'].interpolate(method='polynomial', order=orders)
            data_feature['count'] = data_feature['count'].replace(0, np.nan)
            data_feature['count'] = data_feature['count'].interpolate(method='polynomial', order=orders)
            data_feature['std'] = data_feature['std'].interpolate(method='polynomial', order=orders)
            data_feature['min'] = data_feature['min'].interpolate(method='polynomial', order=orders)
            data_feature['max'] = data_feature['max'].interpolate(method='polynomial', order=orders)
            data_feature['q1'] = data_feature['q1'].interpolate(method='polynomial', order=orders)
            data_feature['q2'] = data_feature['q2'].interpolate(method='polynomial', order=orders)
            data_feature['q3'] = data_feature['q3'].interpolate(method='polynomial', order=orders)
            # data_feature['skew'] = data_feature['skew'].interpolate(method='polynomial', order=orders)

            try:
                data_feature['skew'] = data_feature['skew'].interpolate(method='polynomial', order=orders)
            except Exception as err:
                data_feature['skew'] = data_feature['skew'].fillna(0)

            data_feature['std'] = data_feature['std'].fillna(0)
            data_feature['skew'] = data_feature['skew'].fillna(0)

            data_feature = data_feature.reset_index()
            print('-- clean --')
            print(data_feature)

            def create_epochten(date_time):
                date_time = datetime.strptime(str(date_time), '%Y-%m-%d %H:%M:%S')
                epoch = date_time.replace(tzinfo=timezone.utc).timestamp()
                return int(epoch)

            data_feature['epoch_ten'] = data_feature.apply(lambda x: create_epochten(x['utc']), axis=1)
            data_feature['name'] = tmname.upper()
            data_feature.index = data_feature.index + last_id_ana
            print('-- feature --')
            print(data_feature)
        except Exception as err:
            gf.write_filelogging(
                'warning', 
                'Feature calculation of {} with {} is error in create feature process | Error: {}'.
                format(tmname, freq, err))
            return

    # record result to analysis table
    # data_feature = pd.DataFrame()
    if data_feature.empty == True:
        gf.write_filelogging(
            'warning', 
            'There is something wrong in feature calculating for {} with {} in create feature process| The data is loss'.
            format(tmname, freq))
    else:
        # print(f'--- record feature result {tmname}---')
        try:
            delete_latestrow_analysisTMtable_sql = '''DELETE FROM {} WHERE id >= {}'''.format(
                feature_table, last_id_ana)
            st.cursor_tmanalysis.execute(delete_latestrow_analysisTMtable_sql)
            st.connect_tmanalysis.commit()
        except Exception as err:
            gf.write_filelogging(
                'warning', 'Delete latest row of {} is error in create feature process | Error: {}'.format(feature_table, err))
            return
        else:
            try:
                gf.record_buffer(
                    st.connect_tmanalysis, st.cursor_tmanalysis, feature_table, data_feature)
                st.active_tm_createfeature.append(feature_table)
                st.detectanomaly_state = True
                print(" finish detectanomaly_state ----> ", st.detectanomaly_state)
            except Exception as err:
                gf.write_filelogging(
                    'warning', 'Feature data recording to {} table is error in create feature process | Error: {}'.format(feature_table, err))
                return


def detect_tmanomaly(featuretable):
    print("Start Detect anomaly ")
    downloadmodel = False
    checkpoint_model = False
    # checkpoint_complete = False
    # checkpoint_complete = False
    transformtensor_complete = False
    loadmodel_complete = False
    anomalydetection_complete = False

    tm_name = featuretable.split('_')[2].upper()
    freq = featuretable.split('_')[3].upper()

    info_analysis_sql = """SELECT transform_method, algorithm_name, model_address, model_name, anomaly_result_table
                        FROM analysis_info_theos_auto_m1 WHERE feature_table = \'{}\';""".format(featuretable)
    info_analysis = gf.read_sql(info_analysis_sql, conn=st.connect_tmanalysis)
    # print(info_analysis)

    if info_analysis.empty == True:
        gf.write_filelogging(
            'warning', 'Cannot retrive the data of {} table in detect anomaly process | Cannot retrive data'.format(featuretable))
    else:
        # transform_method = info_analysis['transform_method'][0]
        model_addess = info_analysis['model_address'][0]
        model_name = info_analysis['model_name'][0]
        algorithm_name = info_analysis['algorithm_name'][0]
        anomaly_table = info_analysis['anomaly_result_table'][0]
        remote_model = "".join([model_addess, model_name])
        downloadmodel = True

    # Download model from FTP server
    # if os.path.exists(st.model_path):
    #     os.remove(st.model_path)
    # if downloadmodel == True:
    #     try:
    #         with open(st.model_path, "wb") as file:
    #             st.ftp.retrbinary('RETR {}'.format(remote_model), file.write)
    #             checkpoint_model = True
    #             # print('Download model complete')
    #     except Exception as err:
    #         gf.write_filelogging(
    #             'error', 
    #             'Cannot download {} model from MMGS server in detect anomaly process | Error: {}'.
    #             format(model_name, err))
            
    # Download model from localhost server
    # if os.path.exists(st.model_path):
    #     os.remove(st.model_path)
    # if downloadmodel == True:
    #     try:
    #         with open(st.model_path, "wb") as file:
    #             file.write(st.model_archivelocal_path)
    #             checkpoint_model = True
    #     except Exception as err:
    #         gf.write_filelogging(
    #             'error', 
    #             'Cannot download {} model from localhost server in detect anomaly process | Error: {}'.
    #             format(model_name, err))

    # Check model is exist in archive model directory
    modelpt_path = os.path.join(st.model_archivelocal_path, model_name)
    if not os.path.exists(modelpt_path):
        gf.write_filelogging('warning', 'The model file ({}) is not located within the archive model directory'.format(model_name))
    else:    
        checkpoint_model = True
        
    # Completeness check for downloaded model
    if checkpoint_model == True:
        try:
            checkpoint = torch.load(modelpt_path)
            look_back = checkpoint['input']
            model_state_dict = checkpoint['state_dict']
            # ewma_mean = checkpoint['ewma_params'][0]
            # ewma_std = checkpoint['ewma_params'][1]
            anomaly_values = checkpoint['anomaly_value']
            transform_method_name = checkpoint['transform_method'][0]
            transform_method_params = checkpoint['transform_method'][1]
            # checkpoint_complete = True
        except Exception as err:
            gf.write_filelogging(
                'warning', 
                'Cannot retrive the data from autom1_model.pt ({}) in detect anomaly process | Error: {}'.
                format(model_name, err))

    # if checkpoint_complete == False:
    #     gf.write_filelogging(
    #         'warning', '{} error: Data may have been lost in detect anomaly process | '.format(model_name))
        else:
            idlast_anomaly_sql = """SELECT max(id) as last_id_ano FROM {};""".format(anomaly_table)
            idlast_anomaly = gf.read_sql(idlast_anomaly_sql, conn=st.connect_tmanalysis)
            id_callanalysis = idlast_anomaly['last_id_ano'][0]-500

            # load feature table
            feature_df_sql = """SELECT id, name, utc, epoch_ten, avg FROM {} WHERE id >= {}""".format(featuretable, id_callanalysis)
            feature_df = gf.read_sql(feature_df_sql, conn=st.connect_tmanalysis)

    if feature_df.empty == True:
        gf.write_filelogging(
            'warning', 'Cannot retrive tha data from {} table in detect anomaly process | Cannot retrive data'.format(featuretable))
    else:
        try:
            # convert dataframe to array
            dataset = feature_df['avg'].values.astype('float32')

            if transform_method_name == 'normalization':
                min_transform = transform_method_params[0]
                max_transform = transform_method_params[1]
                scalar = max_transform-min_transform
                dataset = list(
                    map(lambda x: (x-min_transform)/scalar, dataset))

            # rolling window
            dataX = []
            for i in range(len(dataset)-look_back):
                a = dataset[i:(i+look_back)]
                dataX.append(list(a))
            dataX = np.array(dataX)
            # reshape
            dataX = dataX.reshape(-1, 1, look_back)
            # convert numpy.ndarray to Tensor(float32)
            # Torch>1.11.0
            dataX = torch.tensor(dataX, dtype=torch.float32)
            # Torch < 1.8.0
            # dataX = torch.from_numpy(dataX).float()
            # print('---- tensor ----')
            # print(dataX)
            # print(type(dataX))
            transformtensor_complete = True
        except Exception as err:
            gf.write_filelogging(
                'warning', 
                'There is something wrong for Tensor transformation of {} with {} in detect anomaly process | Transform data error'.
                format(tm_name, freq))

    if transformtensor_complete == True:
        try:
            if algorithm_name == 'model_auto_m1':
                sys.path.append(st.models_dirpath)
                import model_auto_m1
                model = model_auto_m1.autoencoder(look_back)
                # load model
                model.load_state_dict(model_state_dict)
                loadmodel_complete = True
        except Exception as err:
            gf.write_filelogging(
                'warning', 
                'There is something wrong for load model of {} with {} in detect anomaly process | Error: {}'.
                format(tm_name, freq, err))

    if loadmodel_complete == True:
        try:
            var_data = Variable(dataX)
            pred_data = model(var_data)
            pred_data = pred_data.view(-1, look_back).cpu().detach().numpy()

            # Reconstuct dataX and pred_data to become real and reco
            # Create real data by using dataX
            # Torch<1.8.0
            # pred_data2 = pred_data.view(-1, look_back).data.cpu().numpy()
            # Torch > 1.11.0
            data = dataX.view(-1, look_back).cpu().detach().numpy()
            real = np.append(data[:, 1], data[len(data)-1])

            # Create reconsturct data (reco) by using pred_data
            reco = np.append(pred_data[:, 1], pred_data[len(pred_data)-1])
            reco = np.insert(reco, 0, reco[0])
            reco = np.delete(reco, len(reco)-1)

            # Create score (diff)
            diff = np.abs(np.subtract(real, reco))
            datascore = {'real': real, 'score': diff}
            anomalystate_dataframe = pd.DataFrame(data=datascore)
            anomalystate_dataframe['anomaly_state'] = anomalystate_dataframe.apply(lambda x: 0 if x['score'] < anomaly_values[0] else (
                1 if x['score'] < anomaly_values[1] else (2 if x['score'] < anomaly_values[2] else 3)), axis=1)
            anomalystate_dataframe['id'] = anomalystate_dataframe.index

            # create result anomaly dateframe
            anomalystate_list = anomalystate_dataframe['anomaly_state'].tolist(
            )
            anomalystate_list = [0]+anomalystate_list
            del anomalystate_list[-1]
            resultanomaly_dataframe = feature_df.copy()
            # feature_df['anomaly_state_auto_m1'] = anomalystate_list
            # resultanomaly_dataframe = feature_df.copy()
            feature_df = None
            # resultanomaly_dataframe['anomaly_state'] = 90
            resultanomaly_dataframe['anomaly_state_auto_m1'] = anomalystate_list
            # print(resultanomaly_dataframe[resultanomaly_dataframe['anomaly_state_auto_m1'] > 0])
            print('---========-----======-----======-----======')
            # print(resultanomaly_dataframe)
            # print(feature_df)

            # gf.show_graphanomalydetection(
            #     real, reco, diff, anomaly_values, anomalystate_dataframe, resultanomaly_dataframe)
            anomalydetection_complete = True

        except Exception as err:
            # print('Error : ', e)
            gf.write_filelogging(
                'warning', 
                'There is something wrong for prediction of {} with {} in detect anomaly process | Error: {}'.
                format(tm_name, freq, err))
    if anomalydetection_complete == True:
        try:
            print('last id ==> ', idlast_anomaly['last_id_ano'][0])
            # delete 10(look_back) row for remove previous error predict
            id_minuslookback = idlast_anomaly['last_id_ano'][0] - look_back
            delete_minuslookback_sql = """DELETE FROM {} WHERE id >= {}""".format(
                anomaly_table, id_minuslookback)
            print(delete_minuslookback_sql)
            st.cursor_tmanalysis.execute(delete_minuslookback_sql)
            st.connect_tmanalysis.commit()

            # select resultanomaly_dataframe while id more than id_minuslookback
            result_dataframe = resultanomaly_dataframe[resultanomaly_dataframe['id']
                                                       >= id_minuslookback]
            resultanomaly_dataframe = None
            result_dataframe.index = result_dataframe['id']
            result_dataframe = result_dataframe.drop(['id'], axis=1)
            print(result_dataframe)
        except Exception as err:
            gf.write_filelogging(
                'warning', 
                'There is something wrong for anomaly dataframe preparing of {} with {} in detect anomaly process | Error: {}'.
                format(tm_name, freq, err))
        else:
            # record result anomaly detection to anomaly_theos_{tm}_{freq} table
            gf.record_buffer(
                st.connect_tmanalysis, st.cursor_tmanalysis, anomaly_table, result_dataframe)
            st.active_tm_anomalydetection.append(anomaly_table)
            st.countdaily_state = True
            print('--- detection complete ---')


def count_anomalydaily(tm_ano):

    insert_countinfo_state = False
    update_latestcountdaily_state = False

    # check exists tm in count_info_theos_auto_m1
    countinfo_sql = """SELECT * FROM count_info_theos_auto_m1 WHERE anomaly_table = \'{}\'; """.format(tm_ano)
    countinfo = gf.read_sql(countinfo_sql, conn=st.connect_tmanalysis)
    # print('count info = ', countinfo)

    # count dataframe
    if countinfo.empty == True:
        # print('----> <----')
        count_startdate = '2021-07-01 00:00:00'
        countinfo_maxid_sql = """SELECT MAX(id) FROM count_info_theos_auto_m1;"""
        # print(countinfo_maxid_sql)
        countinfo_maxid = gf.read_sql(countinfo_maxid_sql, conn=st.connect_tmanalysis)['max'][0]
        # print('count max id = ', countinfo_maxid)
        if countinfo_maxid == None:  # countinfo_maxid=None mean count_info_theos_auto_m1 is empty (No data)
            countinfo_maxid = 0
        countinfo_id = countinfo_maxid+1
        insert_countinfo_state = True
    else:
        countinfo_id = countinfo['id'][0]
        latest_date_sql = """SELECT MAX(count_date) FROM countdaily_tmanomaly_theos_auto_m1 WHERE count_info_id = {}""".format(countinfo_id)
        count_startdate = gf.read_sql(latest_date_sql, conn=st.connect_tmanalysis)['max'][0]
        print('count_startdate')
        print(count_startdate)
        update_latestcountdaily_state = True

    try:
        # count anomaly
        countdaily_sql = """SELECT id, name, utc, anomaly_state_auto_m1 FROM {} WHERE utc >= \'{}\';""".format(tm_ano, count_startdate)
        countdaily_df = gf.read_sql(countdaily_sql, conn=st.connect_tmanalysis)
        # count anomaly points if anomaly level (anomaly_state_auto_m1) more than 2
        countdaily_df['countamout_per_day'] = countdaily_df.apply(lambda x: 0 if x['anomaly_state_auto_m1'] < 2 else 1, axis=1)
        countdaily_df = countdaily_df.groupby(pd.Grouper(key='utc', freq='1D')).sum()
        countdaily_df['count_info_id'] = countinfo_id
        countdaily_df['anomaly_table'] = tm_ano
        countdaily_df['count_date'] = countdaily_df.index
        countdaily_df = countdaily_df.reset_index()
        countdaily_df = countdaily_df.drop(['utc', 'id', 'anomaly_state_auto_m1'], axis=1)
        countdaily_df = countdaily_df[['count_info_id', 'anomaly_table', 'count_date', 'countamout_per_day']]
    except Exception as err:
        gf.write_filelogging(
            'warning', 
            'There is something wrong to counting number of anomaly points for {} in count anomaly daily process| Error: {}'.
            format(tm_ano, err))

    # create id of count anomaly table
    latest_countdaily_id_sql = """SELECT MAX(id) FROM countdaily_tmanomaly_theos_auto_m1;"""
    latest_countdaily_id = gf.read_sql(latest_countdaily_id_sql, conn=st.connect_tmanalysis)['max'][0]
    print('latest_countdaily_id ===> ', tm_ano, latest_countdaily_id)
    if latest_countdaily_id == None:
        countdaily_df.index += 1
    else:
        countdaily_df.index = countdaily_df.index + latest_countdaily_id + 1

    print(countdaily_df)
    print('update_latestcountdaily_state = ', update_latestcountdaily_state)

    def update_latestcountdaily(errmessenge):
        try:
            update_latestcountdaily_sql = """UPDATE countdaily_tmanomaly_theos_auto_m1 SET countamount_per_day={} WHERE count_info_id = {} AND count_date = \'{}\';""".format(
                countdaily_df['countamout_per_day'][latest_countdaily_id + 1], countdaily_df['count_info_id'][latest_countdaily_id + 1], count_startdate)
            st.cursor_tmanalysis.execute(update_latestcountdaily_sql)
            st.connect_tmanalysis.commit()
        except Exception as err:
            gf.write_filelogging(
                'warning', 
                'Update amount of anomaly points for letest date ({}) is error in count anomaly daily process| Error: {}'.
                format(errmessenge, err))

    if update_latestcountdaily_state == True:
        if len(countdaily_df) == 1:  # There is not new anomaly date
            if countdaily_df['count_date'][latest_countdaily_id + 1] == count_startdate:
                print('There is not new anomaly date')
                print(countdaily_df)
                update_latestcountdaily('There is not new anomaly date')
                return
            else:
                gf.write_filelogging(
                    'warning', 
                    'Cannot update amount of anomaly points for letest date (There is not new anomaly date) becouse of latest date is no match for {} in count anomaly daily process'.
                    format(tm_ano))
                return
        else:  # There is new anomaly date
            print('There is new anomaly date')
            print(countdaily_df)
            update_latestcountdaily('There is new anomaly date')
            # drop row of dataframe while updataed
            countdaily_df.drop([latest_countdaily_id+1], axis=0, inplace=True)
    # print('---- countdaily_df ============')
    # print(countdaily_df)
    # record to countdaily_tmanomaly_theos_auto_m1
    gf.record_buffer(st.connect_tmanalysis, st.cursor_tmanalysis, 'countdaily_tmanomaly_theos_auto_m1', countdaily_df)

    if insert_countinfo_state == True:
        analysis_info_sql = """SELECT id, tm_name, freq, feature_table FROM analysis_info_theos_auto_m1 WHERE anomaly_result_table = \'{}\';""".format(tm_ano)
        analysis_info = gf.read_sql(analysis_info_sql, conn=st.connect_tmanalysis)
        analysis_info_id = analysis_info['id'][0]
        tm_name = analysis_info['tm_name'][0]
        freq = analysis_info['freq'][0]
        feature_table = analysis_info['feature_table'][0]
        print(tm_name)
        print(analysis_info)

        analysis_params_sql = """SELECT id FROM analysis_params_theos_auto_m1 WHERE feature_table = \'{}\';""".format(feature_table)
        print(analysis_params_sql)
        analysis_params_id = gf.read_sql(analysis_params_sql, conn=st.connect_tmanalysis)['id'][0]
        print('analysis_params_id = ', analysis_params_id)

        record_table = 'record_theos_{}'.format(tm_name.lower())
        # print('record_table = ', record_table)

        try:
            countinfo_values = [[tm_name, freq, tm_ano, feature_table, record_table, analysis_params_id, analysis_info_id]]
            print('countinfo_values = ', countinfo_values)
            count_columns = ['tm_name', 'freq', 'anomaly_table', 'feature_table', 'record_table', 'analysis_params_id', 'analysis_info_id']
            print('count_columns = ', count_columns)
            countinfo_df = pd.DataFrame(countinfo_values, index=[countinfo_id], columns=count_columns)

        except Exception as err:
            gf.write_filelogging(
                'warning', 
                'There is something wrong for insert data to count_info_theos_auto_m1 table of {} in count anomaly process | Error: {}'.
                format(tm_ano, err))

        gf.record_buffer(st.connect_tmanalysis, st.cursor_tmanalysis, 'count_info_theos_auto_m1', countinfo_df)

    print('----== count {} complete ==----'.format(tm_ano))


if __name__ == "__main__":
    rundaily = True
    # if loop_onetime = False is infinte loop
    # if loop_onetime = Ture is one time loop
    loop_onetime = True

    while rundaily:
        st.init()

        rundaily = False

        # delrow.delete_datafortest()
        
        try:
            # st.prepare_tmrecord_state = False
            prepare_preprocessing()

            # with open('tm_active.txt', 'a') as f:
            #     f.write('\n===========================================================\n')
            #     f.write('\n  active tm record = ', st.active_tm_record)
            if st.prepare_tmrecord_state == True:
                prepare_tmrecord()

            # Create Feature
            print("--st.createfeature_state-->", st.createfeature_state)
            if st.createfeature_state == False:
                # st.createfeature_state = False and st.active_tm = [] (or st.createfeature_state = False) mean than the prepare tmrecord porcess for all tms is fails
                # st.createfeature_state = True and st.active_tm = [] mean than the prepare tmrecord porcess is complete but all tms are not updated
                gf.write_filelogging('warning', 'The prepare tmrecord process for all TMs is fails | Process fails')
                
            else:
                if st.active_tm == []:
                    gf.write_filelogging(
                        'warning', 'There are not some telemetries where are update in MIXER-II | Process fails')

                elif st.active_tm != []:
                    # print('=== process start create feature ===')
                    # gf.write_filelogging('info', 'Prepare raw data process is complete')
                    for tmname in st.active_tm:
                        try:
                            analysis_info_sql = """SELECT tm_name, freq, feature_table FROM analysis_info_theos_auto_m1 
                                    WHERE id > 15 and tm_name = \'{}\'; """.format(tmname)
                            # print(analysis_info_sql)
                            info_data = gf.read_sql(
                                analysis_info_sql, st.connect_tmanalysis)
                            print('=====st.active_tm====')
                            print(info_data)
                            if info_data.empty == True:
                                gf.write_filelogging(
                                    'warning', 
                                    'Cannot retrive the data for {} in analysis_info_theos_auto_m1 table | Process fails'.
                                    format(tmname))
                            else:
                                for _, tminfo in info_data.iterrows():
                                    try:
                                        print('----> ',tmname, tminfo['freq'], tminfo['feature_table'])
                                        create_feature(tmname, tminfo['freq'], tminfo['feature_table'])
                                    except:
                                        # gf.write_filelogging('warning', 'Create feature precess Error for {} with {}'.format(tmname, tminfo['freq']))
                                        continue
                        except:
                            gf.write_filelogging(
                                'warning', 
                                'Create feature precess Error: Cannot retrive the data of {} in analysis_info_theos_auto_m1 table | Process fails'.
                                format(tmname))
                            continue
            # with open('tm_active.txt', 'a') as f:
            #     f.write('\n  active tm feature = ', st.active_tm_createfeature)


                # Detect Anomaly
            if st.detectanomaly_state == False:
                gf.write_filelogging('warning', 'The create feture process for all TMs is fails | Process fails')
            else:
                if st.active_tm_createfeature == []:
                    gf.write_filelogging('warning', 'Create feature process is error for all telemetries: {} | Process fails'.format(st.active_tm))
                elif st.active_tm_createfeature != []:
                    print("--- st.active_tm_createfeature -->", st.active_tm_createfeature)
                    # gf.write_filelogging('info', 'Create feature process is complete')
                    # mmgsconn_state = False

                    # try:
                    #     # connect to server
                    #     conffig = gf.setup_service('MMGS_server1_server')
                    #     st.ftp = FTP(conffig.hosts)
                    #     st.mmgsserver1_state = 1
                    #     st.ftp.login(user=conffig.users, passwd=conffig.passwords)
                    #     gf.write_filelogging('info', 'Connect to MMGS server success | INFO')
                    #     mmgsconn_state = True
                    # except Exception as err:
                    #     gf.write_filelogging('error', 'Failed to connect to MMGS server | Precess fails: {}'.format(err))

                    # if mmgsconn_state == True:
                    print('\n\n ----- Detection -----\n\n')
                    for featuretable in st.active_tm_createfeature:
                        print(f'\n ----- {featuretable} -----')
                        try:
                            detect_tmanomaly(featuretable)
                        except:
                            continue
                    print('-- after detection ---')
                    print(st.active_tm_anomalydetection)
            # with open('tm_active.txt', 'a') as f:
            #     f.write('\n  active tm anomaly = ', st.active_tm_anomalydetection)

            # Count dailyrun
            if st.countdaily_state == False:
                gf.write_filelogging('warning', 'The create feture process for all TMs is fails | Process fails')
            else:
                if st.active_tm_anomalydetection == []:
                    gf.write_filelogging('warning', 'Anomaly detection process is error for all telemetries: {} | Process fails'.format(st.active_tm_createfeature))
                elif st.active_tm_anomalydetection != []:
                    for tm_ano in st.active_tm_anomalydetection:
                        try:
                            count_anomalydaily(tm_ano)
                        except:
                            continue
                        

            gf.close_database_server()

        except:
            gf.write_filelogging('error', 'The process cannot be executed until complete | Process fails')

            gf.close_database_server()

        # Time Counter
        # return memory
            st.init()
        if loop_onetime == True:
            rundaily = False
        else:
            # run tomorrow at 1 am.
            now = datetime.today()
            tomorrow = now + timedelta(days=1)
            # change hour:minute:second.microsec of tomorrow = 01:00:00.00
            tomorrow = tomorrow.replace(hour=1, minute=0, second=0, microsecond=0)       
            # calculate diffent seconds between prasent date and tomorrow at 1am.
            delay_sec = (tomorrow-now).total_seconds()
            # sleep
            time.sleep(delay_sec)
