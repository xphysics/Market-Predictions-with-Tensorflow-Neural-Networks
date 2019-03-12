import numpy as np
import pandas as pd
import shutil
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error as calculate_mse
import itertools
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# Import DFF model
from neuralnet.models import DeepFeedForward as dff



class TimeSeries(object):
    # Time Series Neural Net Prediction

    def __init__(self):

        self.dataframe = pd.read_csv(os.getcwd() +'/dataset.csv',header=0,index_col=0)

        self.dff_models = {}
        self.rnn_models = {}
        self.lstm_models = {}



    def initialize_dff_model(self,target_symbol,prediction_target,prediction_time=1,neurons_architecture=None,
                            train_initialized_model=True,save_model=True,dataframe=None):
        if target_symbol not in self.dff_models.keys():
            self.dff_models[target_symbol] = {}
        if dataframe is None:
            self.dff_models[target_symbol][prediction_target] = dff(target_symbol,prediction_target, prediction_time,
                                                                    dataframe=self.dataframe.symbol_data[target_symbol],
                                                                    neurons_architecture=neurons_architecture)
        if type(dataframe) == pd.DataFrame:
            self.dff_models[target_symbol][prediction_target] = dff(target_symbol,prediction_target, prediction_time,
                                                                    dataframe=dataframe,
                                                                    neurons_architecture=neurons_architecture)
        if train_initialized_model is True:
            self.dff_models[target_symbol][prediction_target].train(interactive=True,save_model=save_model)

    def model_mse_report(self,target_symbol,prediction_target,return_report=True,verbose=True):
        prediction = self.dff_models[target_symbol][prediction_target].predict(1,True)

        unscaled_mse = calculate_mse(prediction['unscaled'].dropna()['test'],prediction['unscaled'].dropna()['prediction'])
        scaled_mse = calculate_mse(prediction['scaled'].dropna()['test'],prediction['scaled'].dropna()['prediction'])

        if verbose is True:
            model_name = target_symbol + '_' + prediction_target
            print('---< '+model_name+'\'s Prediction MSE >---')
            print('Unscaled:',str(unscaled_mse),'  Scaled:',str(scaled_mse))
        if return_report is True:
            return {'unscaled':unscaled_mse,
                    'scaled':scaled_mse}

    def tune_dff_architecture(self,target_symbol,prediction_target):
        # seek for optimal neurons architecture for the most accurate prediction/more accurate one
        # powers of 2 optimizes memory processing

        print('---< Tuning',target_symbol+'\'s',prediction_target,'DFF Prediction Model >---')
        neurons_counts = [32,64,128,256,512,1024]

        current_architecture = self.dff_models[target_symbol][prediction_target].neurons_architecture

        # maximum hidden layers = 4 , there's no reason to go above that yet
        layer_counts = [2,3,4]

        # Tuning architectures with the objective of MINIMIZING MSE (scaled or unscaled)
        sampled_architectures = []
        sampled_architectures.append(current_architecture)

        all_scaled_mse = []
        all_unscaled_mse = []
        
        for n in layer_counts:
            # all possible permutations for neurons combinations of n layers
            all_permutations = [p for p in itertools.product(neurons_counts, repeat=n)]
            for architecture in all_permutations:
                print('||----> Intializing New Architecture:',str(architecture))
                self.initialize_dff_model(target_symbol,prediction_target,1,list(architecture),train_initialized_model=True,save_model=True)
                mse_report = self.model_mse_report(target_symbol,prediction_target,True,False)
                print('Unscaled MSE:',str(mse_report['unscaled']),'  Scaled MSE:',str(mse_report['scaled']))

                # TEMPORARY: trying to save space by deleting the directory everytime we got our info
                shutil.rmtree(self.dff_models[target_symbol][prediction_target].spec_dir)

                sampled_architectures.append(architecture)
                all_scaled_mse.append(mse_report['scaled'])
                all_unscaled_mse.append(mse_report['unscaled'])

        result = {'architecture': pd.Series(data = sampled_architectures),
                    'unscaled_mse': pd.Series(data = all_unscaled_mse),
                    'scaled_mse': pd.Series(data = all_scaled_mse)}
        
        return pd.DataFrame(result)

    def grab_passing_models(self,mse_threshold):
        result = []
        for i in self.dff_models:
            model = self.dff_models[i]
            if model.mse[-1] <= mse_threshold:
                result.append(i)
        return result           
            


    def describe_predictions(self,target_symbol,prediction_target,mode):
        predictions = self.concat_target_predictions(target_symbol,prediction_target,mode)
        target = target_symbol+'_'+prediction_target+'_'
        last_known_data = predictions[target+'test'].dropna().iloc[-1]
        predictions_indexes = predictions.tail(60)[target+'test'].drop(predictions.tail(60)[target+'test'].dropna().index).index
        
        pred_data = predictions.loc[predictions_indexes]
        non_nans= []
        for c in pred_data.columns:
            for i in pred_data.index:
                if np.isnan(pred_data.at[i,c]):
                    pass
                else:
                    non_nans.append(pred_data.at[i,c])
        
        pred_desc = pd.Series(data=non_nans).describe()

        return pred_desc,last_known_data


    def concat_target_predictions(self,target_symbol,prediction_target,mode):
        # target_model format ex: AAPL_ema13 
        # This will concat all prediction timeframes of the target model into one
        predictions = {}
        for target in self.dff_models:
            if (target_symbol in target) and (prediction_target in target):
                model_prediction = pd.DataFrame().append(self.dff_models[target].prediction)
                missing_indexes = model_prediction.test.drop(model_prediction.test.dropna().index).index
                # Merging takes the difference and apply it to test data
                if (mode == 'raw') or (mode == 'mean_raw'):
                    predictions[target] = model_prediction.prediction
                if (mode == 'merge') or (mode == 'mean_merge'):
                    delta = (model_prediction.prediction - model_prediction.prediction.shift(1)).loc[missing_indexes]
                    for stamp in delta.index:
                        previous_stamp_loc = model_prediction.index.get_loc(stamp) - 1
                        model_prediction.at[stamp,'test'] = model_prediction.at[list(model_prediction.index)[previous_stamp_loc],'test'] + delta.loc[stamp]
                    predictions[target] = model_prediction.test
        # Add a final column of test values

        all_times = []
        target = target_symbol+'_'+prediction_target+'_'
        for t in self.dff_models:
            if target in t:
                pred_time = t.split('_')[-1]
                if pred_time.isdigit():
                    all_times.append(int(pred_time))
        if len(all_times) == 0:
            longest_prediction_key = target+str(max(self.prediction_times))
        else:
            max_loc = all_times.index(max(all_times))
            longest_prediction_key = target+str(all_times[max_loc])
        
        predictions[target_symbol+'_'+prediction_target+'_test'] = self.dff_models[longest_prediction_key].prediction.test
        known_indexes = predictions[target_symbol+'_'+prediction_target+'_test'].dropna().index
        # We'll also all the close prices that we have

        close_test = self.dff_models[longest_prediction_key].dataframe.latest_symbol_data[target_symbol].close.loc[known_indexes]
        predictions[target_symbol+'_close_test'] = close_test
        predictions = pd.DataFrame(predictions)
        if (mode == 'mean_raw') or (mode == 'mean_merge'):
            new_predictions = {}
            for i in predictions.columns:
                new_predictions[i] = predictions.loc[predictions[i].dropna().index].mean(axis=1)
            predictions = pd.DataFrame(new_predictions)
        return predictions

    
    def validate_predictions(self,models=None):
        if models is None:
            to_validate = self.dff_models
        else:
            assert (type(models) == list), 'Insert a list of models'
            to_validate = models
        for s in to_validate:
            self.dff_models[s].validate_prediction()

    def update_predictions(self,models=None,train_before_predict=True,train_mse_threshold=None,training_epoch=None):
        if models is None:
            to_update = self.dff_models
        else:
            assert (type(models) == list), 'Insert a list of models'
            to_update = models
        for s in to_update:
            self.dff_models[s].update_prediction(train_before_predict=train_before_predict,train_mse_threshold=train_mse_threshold,training_epoch=training_epoch)
    

    def hp_trend_report(self,series):
        # categorize trend in ordered time using HP Filter's derivative (dtrend (delta trend))
        # RETURN: Sequential list of trends with respect to time
        #         Each element contains: [direction, extracted dataframe based on trend's index (this will be used for multiple other purposes)]
        # IMPORTANT: THIS CAN ONLY BE APPLIED TO STATIC DATAFRAME, USE FOR HISTORICAL DATA ANALYSIS
        #            WITH ABSOLUTE LOOK-AHEAD BIAS. RECOMMEND TO USE PERIODICALLY (DAILY, HOURLY, ETC)
        hpfilter = sm.tsa.filters.hpfilter(series)
        hp_trend = hpfilter[1]
        hp_noise = hpfilter[0]
        result = pd.DataFrame(series)
        result['hptrend'] = hp_trend
        result['hpnoise'] = hp_noise
        dtrend = ((hp_trend/hp_trend.shift(1))-1).dropna()
        start = 0
        record = []
        track = []
        while len(track) != len(dtrend.index):
            if dtrend[start] < 0:
                down = (dtrend < 0)
                indexes = []
                for i in range(start,len(down)):
                    if down[i] == True:
                        track.append(i)
                        indexes.append(down.index[i])
                        if i == range(len(down))[-1]: #if we reach last value
                            record.append(['DOWN',result.loc[pd.Index(indexes)]])
                            break
                    elif down[i] == False:
                        start = i
                        record.append(['DOWN',result.loc[pd.Index(indexes)]])
                        break
            elif dtrend[start] > 0:
                up = (dtrend > 0)
                indexes = []
                for i in range(start,len(up)):
                    if up[i] == True:
                        track.append(i)
                        indexes.append(up.index[i])
                        if i == range(len(up))[-1]: #if we reach last value
                            record.append(['UP',result.loc[pd.Index(indexes)]])
                            break
                    elif up[i] == False:
                        start = i
                        record.append(['UP',result.loc[pd.Index(indexes)]])
                        break
            elif dtrend[start] == 0:
                still = (dtrend == 0)
                indexes = []
                for i in range(start,len(still)):
                    if still[i] == True:
                        track.append(i)
                        indexes.append(still.index[i])
                        if i == range(len(still))[-1]: #if we reach last value
                            record.append(['STILL',result.loc[pd.Index(indexes)]])
                            break
                    elif still[i] == False:
                        start = i
                        record.append(['STILL',result.loc[pd.Index(indexes)]])
                        break
        return record
