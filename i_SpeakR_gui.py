#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 13:37:46 2022

@author: Mrinmoy Bhattacharjee, Senior Project Engineer, Dept. of EE, IIT Dharwad
"""

REDIRECT_OUTPUT = True
import os
os.environ['FLASK_APP'] = 'i_SpeakR_gui'
os.environ["FLASK_RUN_PORT"] = '443'
from flask import url_for, request, Flask, render_template
import webbrowser
import datetime
from lib.data_io.metafile_reader import GetMetaInfo
from lib.data_io.chop_utterances import ChopUtterances
from lib.feature_computation.compute_mfcc import MFCC
from lib.feature_computation.load_features import LoadFeatures
from lib.models.GMM_UBM.gaussian_background import GaussianBackground
import numpy as np
import pickle
import sys
if os.path.exists('static/output.txt'):
    os.remove('static/output.txt')
if REDIRECT_OUTPUT:
    sys.stdout.close()
    sys.stdout = open('static/output.txt', 'w')
from multiprocessing import Process
from werkzeug.utils import secure_filename
import shutil

data_type_ = []
result_filename_ = ''

def open_browser():
    webbrowser.open('https://127.0.0.1:5000')

app = Flask(__name__)
output_path = os.getcwd() + '/../i-SpeakR_GUI_output/'
if not os.path.exists(output_path):
    os.makedirs(output_path)
UPLOAD_FOLDER = output_path + '/data/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
MAX_CONTENT_PATH = 53687091200
data_path_ = ''



@app.route('/')
def landing_page():
    if REDIRECT_OUTPUT:
        sys.stdout = open('static/output.txt', 'w')
    print(f'UPLOAD_FOLDER={UPLOAD_FOLDER}')
    datasets_ = next(os.walk(UPLOAD_FOLDER))[1]
    print(f'datasets_={datasets_}')
    style_url_ = url_for('static', filename='style.css')
    return render_template('data_upload.html', style=style_url_, datasets=datasets_)
        



@app.route('/uploader.html', methods = ['POST'])
def upload_dataset():
    if REDIRECT_OUTPUT:
        sys.stdout = open('static/output.txt', 'a+')
    dataset_name_ = ''
    if request.method == 'POST':
        if request.form['available_datasets']=="other":
            for f_ in request.files.getlist('file[]'):
                path_ = '/'.join(os.path.join(UPLOAD_FOLDER, f_.filename).split('/')[:-1]) + '/'
                if not os.path.exists(path_):
                    os.makedirs(path_)
                print(f"path_={path_} filename={secure_filename(f_.filename.split('/')[-1])}")
                f_.save(os.path.join(path_, f_.filename.split('/')[-1]))
                dataset_name_ = f_.filename.split('/')[0]
                data_path_ = UPLOAD_FOLDER + '/' + dataset_name_ + '/'
            print('Files uploaded successfully')
        else:
            dataset_name_ = request.form['available_datasets']
            data_path_ = UPLOAD_FOLDER + '/' + dataset_name_ + '/'
    
    global data_type_
    data_type_ = []
    
    metaobj_infer_ = GetMetaInfo(data_info="infer", path=data_path_, duration=True, gender=True)
    dev_sets_infer_, enr_sets_infer_, test_sets_infer_ = select_data_sets(metaobj_infer_.INFO, 'infer')
    if ((len(dev_sets_infer_)>=1) and (len(enr_sets_infer_)>=1) and (len(test_sets_infer_)>=1)):
        data_type_.append('infer')

    metaobj_specify_ = GetMetaInfo(data_info="specify", path=data_path_, duration=True, gender=True)
    dev_sets_specify_, enr_sets_specify_, test_sets_specify_ = select_data_sets(metaobj_specify_.INFO, 'specify')
    if ((len(dev_sets_specify_)>=1) and (len(enr_sets_specify_)>=1) and (len(test_sets_specify_)>=1)):
        data_type_.append('specify')    
    
    print('data_type: ', data_type_)
    if data_type_:
        style_url_ = url_for('static', filename='style.css')
        return render_template('setup_experiment.html', style=style_url_, data_path=data_path_, dataset_name=dataset_name_, dev_sets=dev_sets_specify_, enr_sets=enr_sets_specify_, test_sets=test_sets_specify_, data_type=data_type_)
    else:
        style_url_ = url_for('static', filename='style.css')
        return render_template('status.html', style=style_url_, error_msg="Data folder not in correct format. If basic mode is selected, the program expects three folders in <data_path>, viz. DEV, ENR, TEST. All wav files for DEV, ENR or TEST sets needs to be kept within each respective directories without any sub-directories. If advanced mode is selected, searches for DEV*.csv, ENR*.csv and TEST*.csv in <data_path>")





def select_data_sets(info, data_type):
    '''
    Function to select development, enrollment and test sets when multiple 
    options are available.

    Parameters
    ----------
    info : dict
        Dict contains the information about various sets in the dataset.
    data_type : str
        'infer' or 'specify'.

    Returns
    -------
    info : dict
        A pruned dict that contains only those sets that are selected by the
        user.

    '''
    if data_type=='specify':
        end_str_ = '.csv'
    else:
        end_str_ = ''

    # Select set names starting with "DEV"
    dev_sets = [set_name if (set_name.startswith('DEV') and set_name.endswith(end_str_)) else '' for set_name in info.keys()] 
    # Remove empty set names
    dev_sets = list(filter(None, dev_sets))                                                 
    
    # Select set names starting with "ENR"
    enr_sets = [set_name if (set_name.startswith('ENR') and set_name.endswith(end_str_)) else '' for set_name in info.keys()]
    # Remove empty set names
    enr_sets = list(filter(None, enr_sets))
    
    # Select set names starting with "TEST"
    test_sets = [set_name if (set_name.startswith('TEST') and set_name.endswith(end_str_)) else '' for set_name in info.keys()]
    # Remove empty set names
    test_sets = list(filter(None, test_sets))
    
    return dev_sets, enr_sets, test_sets




@app.route('/parse_parameters.html', methods=['POST'])
def parse_parameters(name=None):
    if REDIRECT_OUTPUT:
        sys.stdout = open('static/output.txt', 'a+')
    if request.method == 'POST':
        CFG = {
            'TODAY': datetime.datetime.now().strftime("%Y-%m-%d"),
            'SAMPLING_RATE': int(request.form['sampling_rate']),             # Sampling rate to be used for the audio files
            'NFFT': int(request.form['n_fft']),                              # Number of DFT points to be used
            'FRAME_SIZE': int(request.form['frame_size']),                   # Short-term frame size in miliseconds
            'FRAME_SHIFT': int(request.form['frame_shift']),                 # Short-term frame shift in miliseconds 
            'DEV_CHOP': [50],   # Development sub-utterance sizes
            'ENR_CHOP': [50],     # Enrollment sub-utterance sizes
            'TEST_CHOP': [10,20,30,40,50,60],   # Test sub-utterance sizes
            'PREEMPHASIS': bool(request.form['preemphasis']),           # Boolean flag indicating pre-emphasis required or not. True indicates pre-emphasis is required, False indicates pre-emphasis not required.
            'N_MELS': int(request.form['n_mels']),                           # Number of Mel filters to be used
            'N_MFCC': int(request.form['n_mfcc']),                           # Number of MFCC coefficients to be computed. If EXCL_C0=True, N_MFCC+1 coefficients are computed and c0 is ignored
            'COMPUTE_DELTA_FEAT': bool(request.form['compute_delta_feat']), # Boolean flag indicating whether delta features are computed
            'DELTA_WIN': int(request.form['delta_win']),                     # Context window to be used for computing Delta features
            'EXCL_C0': bool(request.form['excl_c0']),                   # Boolean flag indicating whether MFCC c0 to be used or not. True indicates c0 is included. False indicates c0 is to be ignored and N_MFCC+1 coefficients to be computed
            'FEATURE_NAME': request.form['feature_name'],                    # Parameter to indicate which feature to compute
            'FEATURE_SCALING': int(request.form['feature_scaling']),         # Type of feature scaling to be used.
                                                                        # 0: no scaling, 
                                                                        # 1: only mean subtraction,      
                                                                        # 2: mean and variance scaling
            'MODEL_TYPE': request.form['model'],                             # Parameter indicating which model to use
            'UBM_NCOMPONENTS': int(request.form['UBM_ncomp']),               # Number of Gaussian components for the UBM model
            'COVARIANCE_TYPE': request.form['covariance_type'],              # Type of covariance: 'full', 'diag', 'tied'
            'ADAPT_WEIGHT_COV': bool(request.form['adapt_weight_cov']), # Flag to indicate whether to adapt the weights and covariances of the speaker models
            'DATA_PATH': request.form['data_path'], # Flag to indicate whether to adapt the weights and covariances of the speaker models
            'OUTPUT_PATH': output_path, # request.form['output_path'], # Flag to indicate whether to adapt the weights and covariances of the speaker models
            'DATA_TYPE': request.form['data_type'],                          # Switch to select how to obtain the dataset details
            'DEV_CSV': request.form['dev_csv'],                                # DEV set .csv file
            'ENR_CSV': request.form['enr_csv'],                                # ENR set .csv file
            'TEST_CSV': request.form['test_csv'],                              # TEST set .csv file
            }
        
        if CFG['FEATURE_NAME']=='MFCC':
            if CFG['COMPUTE_DELTA_FEAT']:
                CFG['NUM_DIM'] = 3*CFG['N_MFCC']
            else:
                CFG['NUM_DIM'] = CFG['N_MFCC']
    
        CFG['OUTPUT_DIR'] = CFG['OUTPUT_PATH'] + '/' + CFG['DATA_PATH'].split('/')[-2] + '/'
        if not os.path.exists(CFG['OUTPUT_DIR']):
            os.makedirs(CFG['OUTPUT_DIR'])
            
        CFG['SPLITS_DIR'] = CFG['OUTPUT_DIR'] + '/sub_utterance_info/'
        if not os.path.exists(CFG['SPLITS_DIR']):
            os.makedirs(CFG['SPLITS_DIR'])
    
        CFG['FEAT_DIR'] = CFG['OUTPUT_DIR'] + '/features/' + CFG['FEATURE_NAME'] + '/'
        if not os.path.exists(CFG['FEAT_DIR']):
            os.makedirs(CFG['FEAT_DIR'])
    
        CFG['MODEL_DIR'] = CFG['OUTPUT_DIR'] + '/models/' + CFG['FEATURE_NAME'] + '_' + CFG['MODEL_TYPE'] + '/'
        if not os.path.exists(CFG['MODEL_DIR']):
            os.makedirs(CFG['MODEL_DIR'])
        
        if CFG['DATA_TYPE']=='infer':
            metaobj_ = GetMetaInfo(data_info="infer", path=data_path_, duration=False, gender=False)
        elif CFG['DATA_TYPE']=='specify':
            metaobj_ = GetMetaInfo(data_info="specify", path=data_path_, duration=False, gender=False)
            key_list_ = [key for key in metaobj_.INFO.keys()]
            for key in key_list_:
                if key not in [CFG['DEV_CSV'], CFG['ENR_CSV'], CFG['TEST_CSV']]:
                    del metaobj_.INFO[key]
        
        with open('static/output.txt', 'w+') as f_:
            for key in CFG.keys():
                f_.write(f'{key}=CFG[key]\n')
            f_.write('\n\n\n')
        
        Process(run_toolkit(CFG, metaobj_))

        style_url_ = url_for('static', filename='style.css')
        f_ = open('static/output.txt', 'r')
        status_ = f_.read()
        f_.close()
        return render_template('status.html', style=style_url_, output=status_)
    else:
        return '<p style="margin: auto;">GET method not allowed</p>'



@app.route('/status.html', methods = ['GET', 'POST'])
def get_status():
    if REDIRECT_OUTPUT:
        sys.stdout = open('static/output.txt', 'a+')
    style_url_ = url_for('static', filename='style.css')
    f_ = open('static/output.txt', 'r')
    status_ = f_.read()
    f_.close()
    global result_filename_
    if not result_filename_=='':
        download_url_ = url_for('static', filename=result_filename_+'.zip')
        return render_template('status.html', style=style_url_, output=status_, download_result=download_url_)
    else:
        return render_template('status.html', style=style_url_, output=status_)




def run_toolkit(CFG, metaobj):
    global result_filename_
    result_filename_ = ''
    if REDIRECT_OUTPUT:
        sys.stdout = open('static/output.txt', 'a')
    
    print('Global variables:\n')
    for key in CFG:
        print(f'\t{key}={CFG[key]}\n')
    print('\n\n')

    
    print('Utterance chopping..')
    ChopUtterances(config=CFG).create_splits(metaobj.INFO, CFG['DATA_PATH'])
    print('\n\n')
    

    print('Feature computation..')
    if CFG['FEATURE_NAME']=='MFCC':
        feat_info_ = MFCC(config=CFG).compute(CFG['DATA_PATH'], metaobj.INFO, CFG['SPLITS_DIR'], CFG['FEAT_DIR'], delta=CFG['COMPUTE_DELTA_FEAT'])
        print('\n\n')
            
    if CFG['MODEL_TYPE']=='GMM_UBM':
        GB_ = GaussianBackground(
            model_dir=CFG['MODEL_DIR'], 
            opDir=CFG['OUTPUT_DIR'],
            num_mixtures=CFG['UBM_NCOMPONENTS'], 
            feat_scaling=CFG['FEATURE_SCALING']
            )
        
        '''
        Training the GMM-UBM model
        '''
        ubm_fName = CFG['MODEL_DIR'] + '/ubm.pkl'
        if not os.path.exists(ubm_fName):
            dev_key_ = list(filter(None, [key if key.startswith('DEV') else '' for key in feat_info_.keys()]))
            if not os.path.exists(CFG['OUTPUT_DIR']+'/DEV_Data.pkl'):
                FV_dev_ = LoadFeatures(info=feat_info_[dev_key_[0]], feature_name=CFG['FEATURE_NAME']).load(dim=CFG['NUM_DIM'])
                # with open(CFG['OUTPUT_DIR']+'/DEV_Data.pkl', 'wb') as f_:
                #     pickle.dump(FV_dev_, f_, pickle.HIGHEST_PROTOCOL)
            else:
                with open(CFG['OUTPUT_DIR']+'/DEV_Data.pkl', 'rb') as f_:
                    FV_dev_ = pickle.load(f_)
            GB_.train_ubm(FV_dev_, cov_type=CFG['COVARIANCE_TYPE'])
        else:
            print('The GMM-UBM is already available')
        print('\n\n')
        
        
        ''' 
        Speaker-wise adaptation 
        '''
        enr_key_ = list(filter(None, [key if key.startswith('ENR') else '' for key in feat_info_.keys()]))
        if not os.path.exists(CFG['OUTPUT_DIR']+'/ENR_Data.pkl'):
            FV_enr_ = LoadFeatures(info=feat_info_[enr_key_[0]], feature_name=CFG['FEATURE_NAME']).load(dim=CFG['NUM_DIM'])
            # with open(CFG['OUTPUT_DIR']+'/ENR_Data.pkl', 'wb') as f_:
            #     pickle.dump(FV_enr_, f_, pickle.HIGHEST_PROTOCOL)
        else:
            with open(CFG['OUTPUT_DIR']+'/ENR_Data.pkl', 'rb') as f_:
                FV_enr_ = pickle.load(f_)
                
        GB_.speaker_adaptation(
            FV_enr_, 
            cov_type=CFG['COVARIANCE_TYPE'], 
            use_adapt_w_cov=CFG['ADAPT_WEIGHT_COV']
            )
        print('\n\n')
                
            
        ''' 
        Testing the trained models 
        '''
        test_key_ = list(filter(None, [key if key.startswith('TEST') else '' for key in feat_info_.keys()]))
        test_opDir_ = CFG['OUTPUT_DIR'] + '/' + test_key_[0].split('.')[0] + '/'
        if not os.path.exists(test_opDir_):
            os.makedirs(test_opDir_)
                
        for utter_dur_ in CFG['TEST_CHOP']:
            res_fName = test_opDir_ + '/Result_'+str(utter_dur_)+'s.pkl'
            if not os.path.exists(res_fName):
                '''
                All test-data loaded at-once
                '''
                '''
                FV_test_ = LoadFeatures(info=feat_info_[test_key_[0]], feature_name=CFG['FEATURE_NAME']).load(dim=CFG['NUM_DIM'])
                scores_ = GB_.perform_testing(opDir=CFG['OUTPUT_DIR'], opFileName='Test_Scores', X_TEST=FV_test_, duration=utter_dur_)
                '''
                
                '''
                Utterance-wise testing
                '''
                scores_ = GB_.perform_testing(opDir=test_opDir_, feat_info=feat_info_[test_key_[0]], dim=CFG['NUM_DIM'], duration=utter_dur_)
                
                with open(res_fName, 'wb') as f_:
                    pickle.dump({'scores':scores_}, f_, pickle.HIGHEST_PROTOCOL)
            else:
                with open(res_fName, 'rb') as f_:
                    scores_ = pickle.load(f_)['scores']

            ''' 
            Computing the performance metrics 
            '''
            metrics_ = GB_.evaluate_performance(scores_)
            # roc_opFile = test_opDir_ + '/ROC_' + str(utter_dur_) + 's.png'
            # PerformanceMetrics().plot_roc(metrics_['fpr'], metrics_['tpr'], roc_opFile)
                
            print(f'\n\nUtterance duration: {utter_dur_}s:\n__________________________________________')
            print(f"\tAccuracy: {np.round(metrics_['accuracy']*100,2)}")
            print(f"\tMacro Average Precision: {np.round(metrics_['precision']*100,2)}")
            print(f"\tMacro Average Recall: {np.round(metrics_['recall']*100,2)}")
            print(f"\tMacro Average F1-score: {np.round(metrics_['f1-score']*100,2)}")
            print(f"\tEER: {np.round(np.mean(metrics_['eer'])*100,2)}")

            with open(test_opDir_+'/Performance.txt', 'a+') as f_:
                f_.write(f'Utterance duration: {utter_dur_}s:\n__________________________________________\n')
                f_.write(f"\tAccuracy: {np.round(metrics_['accuracy']*100,2)}\n")
                f_.write(f"\tMacro Average Precision: {np.round(metrics_['precision']*100,2)}\n")
                f_.write(f"\tMacro Average Recall: {np.round(metrics_['recall']*100,2)}\n")
                f_.write(f"\tMacro Average F1-score: {np.round(metrics_['f1-score']*100,2)}\n")
                f_.write(f"\tEER: {np.round(np.mean(metrics_['eer'])*100,2)}\n\n")
    
    if not os.path.exists('results/'):
        os.makedirs('results/')

    path_split_ = list(filter(None, test_opDir_.split('/')))
    result_filename_ = path_split_[-2]+'_'+path_split_[-1]
    if not os.path.exists('static/'+result_filename_+'.zip'):
        shutil.make_archive('static/'+result_filename_, 'zip', test_opDir_)


if __name__ == '__main__':
# 	context = ('flaskssl/1f9476e3959ebe60.crt', 'flaskssl/star_iitdh_key.key')
# 	app.run(host="0.0.0.0", debug=True, port=443, ssl_context=context)
    app.run(host="127.0.0.1", debug=True, port=5000)

    