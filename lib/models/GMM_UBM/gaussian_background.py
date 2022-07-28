#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 29 12:21:17 2022

@author: Mrinmoy Bhattacharjee, Senior Project Engineer, Dept. of EE, IIT Dharwad
"""
from sklearn.mixture import GaussianMixture
import numpy as np
from lib.models.GMM_UBM.speaker_adaptation import SpeakerAdaptation
import pickle
import os
# from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from lib.metrics.performance_metrics import PerformanceMetrics
import psutil
import sys


class GaussianBackground:
    NCOMP = 0
    MODEL_DIR = ''
    BACKGROUND_MODEL = None
    FEATURE_SCALING = 0
    SCALER = None
    N_BATCHES = 0
    
    
    def __init__(self, model_dir, opDir, num_mixtures=128, feat_scaling=0, n_batches=5):
        self.MODEL_DIR = model_dir
        self.OPDIR = opDir
        self.NCOMP = num_mixtures
        self.FEATURE_SCALING = feat_scaling
        self.N_BATCHES = n_batches
    
    
    def train_ubm(self, X, cov_type='diag', max_iterations=100, num_init=1, verbose=1):
        '''
        Training the GMM Universal Background Model.

        Parameters
        ----------
        X : dict
            Dictionary containing the speaker-wise fature arrays.
        cov_type : str, optional
            Type of covariance to be used to train the GMM. The default is 'diag'.
        max_iterations : int, optional
            Maximum number of iterations to train the GMM. The default is 100.
        num_init : int, optional
            Number of random initializations of the GMM model. The default is 3.
        verbose : int, optional
            Flag to indicate whether to print GMM training outputs. The default is 1.

        Returns
        -------
        None.

        '''
        X_combined_ = np.empty([], dtype=np.float32)
        for speaker_id_ in X.keys():
            for split_id_ in X[speaker_id_].keys():
                if np.size(X_combined_)<=1:
                    X_combined_ = X[speaker_id_][split_id_]
                else:
                    X_combined_ = np.append(X_combined_, X[speaker_id_][split_id_], axis=0)
        print(f'X_combined={np.shape(X_combined_)}')

        ''' Feature Scaling '''
        if self.FEATURE_SCALING==1:
            self.SCALER = StandardScaler(with_mean=True, with_std=False).fit(X_combined_)
            X_combined_ = self.SCALER.transform(X_combined_)
        elif self.FEATURE_SCALING==2:
            self.SCALER = StandardScaler(with_mean=True, with_std=True).fit(X_combined_)
            X_combined_ = self.SCALER.transform(X_combined_)
        
        ram_mem_avail_ = psutil.virtual_memory().available >> 20 # in MB; >> 30 in GB
        # feat_memsize_ = sys.getsizeof(X_combined_) >> 20 # in MB; >> 30 in GB
        calculated_var_size_ = np.shape(X_combined_)[0]*np.shape(X_combined_)[1]*self.NCOMP*4 >> 20
        print(f'Available RAM: {ram_mem_avail_}')
        print(f'Feature memory size: {calculated_var_size_}')

        if calculated_var_size_>ram_mem_avail_:
            '''
            Batch-wise training GMM-UBM
            '''
            print('Training GMM-UBM in a batch-wise manner')
            
            batch_size_ = int(np.shape(X_combined_)[0]/self.N_BATCHES)
            random_sample_idx_ = list(range(np.shape(X_combined_)[0]))
            np.random.shuffle(random_sample_idx_)
            batch_start_ = 0
            batch_end_ = 0
            for batch_i_ in range(self.N_BATCHES):
                batch_start_ = batch_end_
                batch_end_ = np.min([batch_start_+batch_size_, np.shape(X_combined_)[0]])
                X_combined_batch_ = X_combined_[random_sample_idx_[batch_start_:batch_end_], :]
                if batch_i_==0:
                    self.BACKGROUND_MODEL = GaussianMixture(n_components=self.NCOMP, covariance_type=cov_type, max_iter=max_iterations, n_init=num_init, verbose=verbose, reg_covar=1e-3)
                    self.BACKGROUND_MODEL.fit(X_combined_batch_)
                    print(f'Batch: {batch_i_+1} model trained')
                else:
                    adapt_ = None
                    del adapt_
                    adapt_ = SpeakerAdaptation().adapt_ubm(X_combined_batch_.T, self.BACKGROUND_MODEL, use_adapt_w_cov=False)
                    self.BACKGROUND_MODEL.means_ = adapt_['means']
                    self.BACKGROUND_MODEL.weights_ = adapt_['weights']
                    self.BACKGROUND_MODEL.covariances_ = adapt_['covariances']
                    self.BACKGROUND_MODEL.precisions_ = adapt_['precisions']
                    self.BACKGROUND_MODEL.precisions_cholesky_ = adapt_['precisions_cholesky']
                    print(f'Batch: {batch_i_+1} model updated')
        
        else:
            self.BACKGROUND_MODEL = GaussianMixture(n_components=self.NCOMP, covariance_type=cov_type, max_iter=max_iterations, n_init=num_init, verbose=verbose)
            self.BACKGROUND_MODEL.fit(X_combined_)
        
        ubm_fName = self.MODEL_DIR + '/ubm.pkl'
        with open(ubm_fName, 'wb') as f:
            pickle.dump({'model':self.BACKGROUND_MODEL, 'scaler':self.SCALER}, f, pickle.HIGHEST_PROTOCOL)
        
        return
    
    
    def speaker_adaptation(self, X_ENR, cov_type='diag', use_adapt_w_cov=True):
        '''
        Adaptation of the UBM model for each enrolling speaker.

        Parameters
        ----------
        X_ENR : dict
            Dictionary containing the speaker-wise enrollment data.
        cov_type : str, optional
            Type of GMM covariance to be used for training. The default is 'diag'.
        use_adapt_w_cov : bool, optional
            Flag indicating whether to use speaker specific covariacne matrix
            during adaptation. The default is True.

        Returns
        -------
        None.

        '''
        if not self.BACKGROUND_MODEL:
            ubm_fName_ = self.MODEL_DIR + '/ubm.pkl'
            if not os.path.exists(ubm_fName_):
                print('Background model does not exist')
                return
            try:
                with open(ubm_fName_, 'rb') as f_:
                    self.BACKGROUND_MODEL = pickle.load(f_)['model']
                with open(ubm_fName_, 'rb') as f_:
                    self.SCALER = pickle.load(f_)['scaler']
            except:
                with open(ubm_fName_, 'rb') as f_:
                    self.BACKGROUND_MODEL = pickle.load(f_)
        

        ''' Feature Scaling '''
        if self.FEATURE_SCALING>0:
            X_combined_ = np.empty([], dtype=np.float32)
            for speaker_id_ in X_ENR.keys():
                for split_id_ in X_ENR[speaker_id_].keys():
                    if np.size(X_combined_)<=1:
                        X_combined_ = X_ENR[speaker_id_][split_id_]
                    else:
                        X_combined_ = np.append(X_combined_, X_ENR[speaker_id_][split_id_], axis=0)
    
            if self.FEATURE_SCALING==1:
                self.SCALER = StandardScaler(with_mean=True, with_std=False).fit(X_combined_)
            elif self.FEATURE_SCALING==2:
                self.SCALER = StandardScaler(with_mean=True, with_std=True).fit(X_combined_)

        # print(f'{X_ENR.keys()}')
        for speaker_id_ in X_ENR.keys():
            speaker_opDir_ = self.MODEL_DIR + '/' + speaker_id_ + '/'
            if not os.path.exists(speaker_opDir_):
                os.makedirs(speaker_opDir_)
            speaker_model_fName_ = speaker_opDir_ + '/' + str(speaker_id_) + '_adapted_model.pkl'
            if os.path.exists(speaker_model_fName_):
                # print(f'Adapted GMM model already available for speaker={speaker_id_}')
                continue
            
            fv_ = None
            del fv_
            fv_ = np.empty([], dtype=np.float32)
            for split_id_ in X_ENR[speaker_id_].keys():
                if np.size(fv_)<=1:
                    fv_ = X_ENR[speaker_id_][split_id_]
                else:
                    fv_ = np.append(fv_, X_ENR[speaker_id_][split_id_], axis=0)
            
            ''' Feature Scaling '''
            if self.FEATURE_SCALING>0:
                fv_ = self.SCALER.transform(fv_)

            adapt_ = SpeakerAdaptation().adapt_ubm(fv_.T, self.BACKGROUND_MODEL, use_adapt_w_cov)
            adapted_gmm_ = None
            adapted_gmm_ = GaussianMixture(n_components=self.NCOMP, covariance_type=cov_type)
            adapted_gmm_.means_ = adapt_['means']
            adapted_gmm_.weights_ = adapt_['weights']
            adapted_gmm_.covariances_ = adapt_['covariances']
            adapted_gmm_.precisions_ = adapt_['precisions']
            adapted_gmm_.precisions_cholesky_ = adapt_['precisions_cholesky']
            
            with open(speaker_model_fName_, 'wb') as f_:
                pickle.dump(adapted_gmm_, f_, pickle.HIGHEST_PROTOCOL)
            print(f'Adapted GMM model saved for speaker={speaker_id_}')
            
            
    def perform_testing(self, opDir, X_TEST=None, feat_info=None, dim=None, duration=None):
        '''
        Compute test speaker scores against all enrollment speaker models.

        Parameters
        ----------
        opDir : str
            Output path.
        X_TEST : dict, optional
            Dictionary containing the speaker-wise test data.
        feat_info : dict, optional
            Dictionary containing info about feature paths.
        dim : int, optional
            Dimension of input feature.
        duration : str, optional
            Selection of which utterance duration to test. Default, tests all
            utterances.

        Returns
        -------
        scores_ : 2D array
            AN (N x N) array consisting of scores for each test speaer against
            each enrollment speaker. N is the number of speakers.

        '''
        if duration:
            score_fName_ = opDir + '/Test_Scores_' + str(duration) + 's.pkl'
        else:
            score_fName_ = opDir + '/Test_Scores.pkl'
            
        if not os.path.exists(score_fName_):
            if not self.BACKGROUND_MODEL:
                ubm_fName_ = self.MODEL_DIR + '/ubm.pkl'
                if not os.path.exists(ubm_fName_):
                    print('Background model does not exist')
                    return
                try:
                    with open(ubm_fName_, 'rb') as f_:
                        self.BACKGROUND_MODEL = pickle.load(f_)['model']
                    with open(ubm_fName_, 'rb') as f_:
                        self.SCALER = pickle.load(f_)['scaler']
                except:
                    with open(ubm_fName_, 'rb') as f_:
                        self.BACKGROUND_MODEL = pickle.load(f_)
            else:
                print('UBM already loaded')
    
            enrolled_speakers_ = next(os.walk(self.MODEL_DIR))[1]
            scores_ = {}
            true_lab_ = []
            pred_lab_ = []

            
            ''' Loading the speaker models '''
            enr_speaker_model_ = {}
            index_ = 0
            for enr_j_ in enrolled_speakers_:
                enr_speaker_opDir_ = self.MODEL_DIR + '/' + enr_j_ + '/'
                enr_speaker_model_fName_ = enr_speaker_opDir_ + '/' + str(enr_j_) + '_adapted_model.pkl'
                if not os.path.exists(enr_speaker_model_fName_):
                    print(f'GMM model does not exist for speaker={enr_j_}')
                    continue
                model_ = None
                with open(enr_speaker_model_fName_, 'rb') as f_:
                    model_ = pickle.load(f_)
                enr_speaker_model_[index_] = {'speaker_id':enr_j_, 'model':model_}
                index_ += 1
                    
            confusion_matrix_ = np.zeros((len(enr_speaker_model_), len(enr_speaker_model_)))
            match_count_ = np.zeros(len(enrolled_speakers_))
            
            if not X_TEST:
                '''
                Testing every test utterance one by one 
                '''
                total_splits_ = 0
                
                speaker_id_list_ = []
                for split_id_ in feat_info.keys():
                    split_dur_ = int(split_id_.split('_')[-2])
                    if duration:
                        if not split_dur_==duration:
                            continue
                    speaker_id_list_.append(feat_info[split_id_]['speaker_id'])
                    total_splits_ += 1

                split_count_ = 0
                for split_id_ in feat_info.keys():
                    '''
                    Checking duration of utterance
                    '''
                    if duration:
                        if not split_id_.split('_')[-2]==str(duration):
                            continue
                    split_count_ += 1
    
                    speaker_id_ = feat_info[split_id_]['speaker_id']
                    feature_path_ = feat_info[split_id_]['file_path']
                    fv_ = None
                    del fv_
                    fv_ = np.load(feature_path_, allow_pickle=True)
                    # The feature vectors must be stored as individual rows in the 2D array
                    if dim:
                        if np.shape(fv_)[0]==dim:
                            fv_ = fv_.T
                    elif np.shape(fv_)[1]>np.shape(fv_)[0]:
                        fv_ = fv_.T
    
                    ''' Feature Scaling '''
                    if self.FEATURE_SCALING>0:
                        fv_ = self.SCALER.transform(fv_)
                        
                    llr_scores_ = np.zeros(len(enr_speaker_model_))
                    matched_speaker_id_ = ''
                    bg_model_scores_ = np.array(self.BACKGROUND_MODEL.score_samples(fv_))
                    for index_i_ in enr_speaker_model_.keys():
                        spk_model_scores_ = np.array(enr_speaker_model_[index_i_]['model'].score_samples(fv_))
                        llr_scores_[index_i_] = np.mean(np.subtract(spk_model_scores_,  bg_model_scores_))
                    map_idx_ = np.argmax(llr_scores_)
                    matched_speaker_id_ = enr_speaker_model_[map_idx_]['speaker_id']
                    match_count_[map_idx_] += 1
                    
                    scores_[split_id_] = {
                        'index': map_idx_,
                        'speaker_id':speaker_id_, 
                        'llr_scores': llr_scores_, 
                        'matched_speaker':matched_speaker_id_,
                        'enrolled_speakers': enrolled_speakers_,
                        }
                    
                    lab_true_ = np.squeeze(np.where(np.array(enrolled_speakers_)==str(speaker_id_)))
                    lab_pred_ = np.squeeze(np.where(np.array(enrolled_speakers_)==str(matched_speaker_id_)))
                    confusion_matrix_[lab_true_, lab_pred_] += 1

                    true_lab_.append(str(speaker_id_))
                    pred_lab_.append(str(matched_speaker_id_))
                    accuracy_ = np.round(np.sum(np.array(true_lab_)==np.array(pred_lab_))/np.size(true_lab_)*100,2)

                    '''
                    Displaying progress
                    '''
                    # sys.stdout.write('\033[2K\033[1G')
                    print(f'\t{split_id_}', end='\t', flush=True)
                    print(f'splits=({split_count_}/{total_splits_})', end='\t', flush=True)
                    print(f'true=({speaker_id_})', end='\t', flush=True)
                    print(f'pred=({matched_speaker_id_})', end='\t', flush=True)
                    print(f'accuracy={accuracy_}%', end='\n', flush=True)
                    
                np.save(opDir+'/Confusion_Matrix_'+str(duration)+'s.npy', confusion_matrix_)


            if not feat_info:
                ''' 
                Testing each sub-utterance with each speaker model 
                '''
                speaker_count_ = 0
                num_speakers_ = len(X_TEST.keys())

                for speaker_id_i_ in X_TEST.keys():
                    speaker_count_ += 1
                    split_count_ = 0
                    num_splits_ = len(X_TEST[speaker_id_i_].keys())
                    for split_id_ in X_TEST[speaker_id_i_].keys():
                        split_count_ += 1
                        
                        '''
                        Checking duration of utterance
                        '''
                        if duration:
                            if not split_id_.split('_')[-2]==str(duration):
                                continue
                        
                        fv_ = None
                        del fv_
                        fv_ = X_TEST[speaker_id_i_][split_id_]
                    
                        ''' Feature Scaling '''
                        if self.FEATURE_SCALING>0:
                            fv_ = self.SCALER.transform(fv_)
                        
                        llr_scores_ = np.zeros(len(enr_speaker_model_))
                        bg_model_scores_ = np.array(self.BACKGROUND_MODEL.score_samples(fv_))
                        for index_i_ in enr_speaker_model_.keys():
                            spk_model_scores_ = np.array(enr_speaker_model_[index_i_]['model'].score_samples(fv_))
                            llr_scores_[index_i_] = np.mean(np.subtract(spk_model_scores_, bg_model_scores_))
                        map_idx_ = np.argmax(llr_scores_)
                        matched_speaker_id_ = enr_speaker_model_[map_idx_]['speaker_id']
                        scores_[split_id_] = {
                            'index': map_idx_,
                            'speaker_id':speaker_id_i_, 
                            'llr_scores': llr_scores_,
                            'matched_speaker':matched_speaker_id_,
                            'enrolled_speakers': enrolled_speakers_,
                            }
                        
                        lab_true_ = np.squeeze(np.where(np.array(enrolled_speakers_)==str(speaker_id_i_)))
                        lab_pred_ = np.squeeze(np.where(np.array(enrolled_speakers_)==str(matched_speaker_id_)))
                        confusion_matrix_[lab_true_, lab_pred_] += 1
                        true_lab_.append(lab_true_)
                        pred_lab_.append(lab_pred_)
                        accuracy_ = np.round(np.sum(np.array(true_lab_)==np.array(pred_lab_))/np.size(true_lab_)*100,2)

                        '''
                        Displaying progress
                        '''
                        sys.stdout.write('\033[2K\033[1G')
                        print(f'\t{split_id_}', end='\t', flush=True)
                        print(f'speakers=({speaker_count_}/{num_speakers_})', end='\t', flush=True)
                        print(f'splits=({split_count_}/{num_splits_})', end='\t', flush=True)
                        print(f'true={speaker_id_i_} ({lab_true_})', end='\t', flush=True)
                        print(f'pred={matched_speaker_id_} ({lab_pred_})', end='\t', flush=True)
                        print(f'accuracy={accuracy_}%', end='\n', flush=True)

                np.save(opDir+'/Confusion_Matrix_'+str(duration)+'s.npy', confusion_matrix_)

            '''
            Saving the scores for the selected duration
            '''
            with open(score_fName_, 'wb') as f_:
                pickle.dump(scores_, f_, pickle.HIGHEST_PROTOCOL)
        else:
            with open(score_fName_, 'rb') as f_:
                scores_ = pickle.load(f_)
        
        return scores_
    
    
    def evaluate_performance(self, res):
        '''
        Compute performance metrics.

        Parameters
        ----------
        res : dict
            Dictionary containing the sub-utterance wise scores.
        opDir : str
            Output path.

        Returns
        -------
        Metrics_ : dict
            Disctionary containg the various evaluation metrics:
                accuracy, precision, recall, f1-score, eer

        '''
        groundtruth_label_ = []
        ptd_labels_ = []
        groundtruth_scores_ = np.empty([])
        predicted_scores_ = np.empty([])
        for split_id_ in res.keys():
            true_speaker_id_ = res[split_id_]['speaker_id']
            pred_speaker_id_ = res[split_id_]['matched_speaker']
                        
            true_label_ = np.squeeze(np.where(np.array(res[split_id_]['enrolled_speakers'])==str(true_speaker_id_)))
            groundtruth_label_.append(true_label_)
            ptd_labels_.append(np.squeeze(np.where(np.array(res[split_id_]['enrolled_speakers'])==str(pred_speaker_id_))))
            gt_score_ = np.zeros((1,np.size(res[split_id_]['enrolled_speakers'])))
            gt_score_[0,true_label_] = 1
            ptd_scores_ = np.zeros((1,np.size(res[split_id_]['enrolled_speakers'])))
            for speaker_id_ in res[split_id_]['enrolled_speakers']:
                lab_ = np.squeeze(np.where(np.array(res[split_id_]['enrolled_speakers'])==str(speaker_id_)))
                ptd_scores_[0,lab_] = res[split_id_]['llr_scores'][lab_]
            
            if np.size(groundtruth_scores_)<=1:
                groundtruth_scores_ = gt_score_
                predicted_scores_ = ptd_scores_
            else:
                groundtruth_scores_ = np.append(groundtruth_scores_, gt_score_, axis=0)
                predicted_scores_ = np.append(predicted_scores_, ptd_scores_, axis=0)
                
        all_speaker_id_ = next(os.walk(self.MODEL_DIR))[1]
        label_list = list(range(np.size(all_speaker_id_)))
        confmat_, precision_, recall_, fscore_ = PerformanceMetrics().compute_identification_performance(groundtruth_label_, ptd_labels_, label_list)
        acc_ = np.sum(np.diag(confmat_))/np.sum(confmat_)

        FPR_, TPR_, EER_, EER_thresh_ = PerformanceMetrics().compute_eer(groundtruth_scores_.flatten(), predicted_scores_.flatten())
                
        Metrics_ = {
            'accuracy': acc_,
            'precision': precision_,
            'recall': recall_,
            'f1-score': fscore_,
            'fpr': FPR_,
            'tpr': TPR_,
            'eer': EER_,
            'eer_threshold': EER_thresh_,
            }
        
        return Metrics_