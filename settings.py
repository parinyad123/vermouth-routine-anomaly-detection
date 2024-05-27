from datetime import date
import rootpath
import os

def init():

    # === path ===
    global root_path, log_path, all_logpath, error_logpath, warn_logpath, models_dirpath, model_path, model_file, model_archivelocal_path
    global config_setup_file, configDB_file, global_function_file, settings_file
    root_path = rootpath.detect()

    log_path = os.path.join(root_path, "log_dir")
    all_logpath = os.path.join(log_path, "all.log")
    warn_logpath = os.path.join(log_path, "warning.log")
    error_logpath = os.path.join(log_path, "error.log")
    models_dirpath = os.path.join(root_path, "models")
    model_path = os.path.join(models_dirpath, "autom1_model.pt")
    model_file = os.path.join(models_dirpath, "model_auto_m1.py")

    config_setup_file = os.path.join(root_path, "config_setup.py")
    configDB_file = os.path.join(root_path, "configDB.ini")
    global_function_file = os.path.join(root_path, "global_function.py")
    settings_file = os.path.join(root_path, "settings.py")

    model_archivelocal_path = os.path.join(os.path.split(root_path)[0], "model_auto_m1", "model")


# === logging ====
    global currentstate, messagelog, errlog
    currentstate = 0
    messagelog = ''
    errlog = ''


# === connect db ====
    global db_ana, db_rec, db_mix2
    db_ana = ''
    db_rec = ''
    db_mix2 = ''

    global connect_tmanalysis, cursor_tmanalysis, connect_tmrecord, cursor_tmrecord, connect_afs_mixer2, cursor_afs_mixer2
    connect_tmanalysis = None
    cursor_tmanalysis = None
    connect_tmrecord = None
    cursor_tmrecord = None
    connect_afs_mixer2 = None
    cursor_afs_mixer2 = None

    global tmanalysis_db_state, tmrecord_db_state, afs_mixer2_state
    tmanalysis_db_state = 0
    tmrecord_db_state = 0
    afs_mixer2_state = 0

# === prepare_preprecessing ===
    global active_tm_record
    active_tm_record = []

# === TM Preparetion ===
    global prepare_tmrecord_state, active_tm
    prepare_tmrecord_state = False # ตัวแปรกลุ่ม state ใช้กรณี func พัง
    active_tm = []

# === create feature ===
    global createfeature_state ,active_tm_createfeature
    createfeature_state = False
    active_tm_createfeature = []

# === model ===
    global ftp, mmgsserver1_state, active_tm_anomalydetection, detectanomaly_state
    ftp = None
    mmgsserver1_state = 0
    active_tm_anomalydetection = []
    detectanomaly_state = False

# === count ===
    global count_startdate, countdaily_state
    count_startdate = '2022-07-01 00:00:00'
    countdaily_state = False
