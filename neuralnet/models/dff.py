import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import floor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error as calculate_mse
import os,os.path
import tensorflow as tf
import time
import shutil





class DeepFeedForward(object):


    def __init__(self,
                model_name,
                prediction_target,
                prediction_time,
                dataframe,
                neurons_architecture = None,
                mse_threshold = 0.00000001):

        # PARAMS:
        # model_name(str) = model's name (ex: 'AAPL's close')
        # prediction_target(str) = the column name we're trying to predict from the dataframe input 
        # prediction_time(int) = amount of shift applied to the prediction column as t time ahead
        # dataframe(pd.DataFrame) = Dataframe of all data needed to train
        # neurons_architecture(list of int) = neurons count with respective layer count (ex: [1024,516,216] <- 3 hidden layers with specified neurons for each layer) 
        # mse_threshold (float) = desired MSE score (optional, without it, training process will run with specified hyper params)

        assert(type(model_name) == str)
        assert(prediction_target in dataframe.columns)
        assert(type(dataframe) == pd.DataFrame)
        assert(type(prediction_time) == int)

        tf.reset_default_graph()
        tf.set_random_seed(1024)

        #----Initialize non-tensorflow variables----#
        if neurons_architecture is None:
            self.neurons_architecture = [512,256,128,64]
        else:
            self.neurons_architecture = neurons_architecture
        self.model_name = model_name
        self.prediction_target = prediction_target
        self.dataframe = dataframe
        # for continuous data stream, set dataframe to another class that's responsible for updating data
        self.input_size = len(self.dataframe.columns)
        self.prediction_time = prediction_time
        self.mse_threshold = mse_threshold
        #------------------------------------------#

        #-------Set Model Directory-------#
        self.spec_dir = './nn_models/dff/'+prediction_target+'_'+str(prediction_time)+'/'+self.model_name
        if os.path.exists(self.spec_dir) == False:
            os.makedirs(self.spec_dir)
        self.model_dir = self.spec_dir+'/'
        for i in range(len(self.neurons_architecture)):
            self.model_dir += str(self.neurons_architecture[i])
            if i != range(len(self.neurons_architecture))[-1]:
                self.model_dir += '-'
        self.model_dir += '/model'
        #-----------INITIALIZE MODEL ARCHITECTURE--------------#
        self.initialize_model_architecture(self.neurons_architecture)
        #-------------------------------------#
        self.scaler = MinMaxScaler()
        self.hyper_params = {'batch_size': 128,
                             'epochs': 5,
                             'test_size': 128}

     
    def train(self,interactive=True,epochs=None,batch_size=None,
                save_model=False,mse_threshold=None,
                shuffle=True,return_all_mse=False):
        #-----------------------LOADING TRAINING PARAMATERS-----------------------#
        #-----------[model made into a class so it can tune its own parameters]
        x_train,y_train,x_test,y_test = self.extract_train_df()

        if epochs is None:
            epochs = self.hyper_params['epochs']
        if batch_size is None:
            batch_size = self.hyper_params['batch_size']
        if mse_threshold is None:
            mse_threshold = self.mse_threshold
        #-------------------------------------------------------------------------#

        if interactive == True:
            plt.ion()
            fig = plt.figure()
            fig.subplots_adjust(hspace=0.5)

            # First Subplot (upper)
            main_ax = fig.add_subplot(211)
            mse_ax = fig.add_subplot(212)
            mse_ax.set_title('MSE Progression')

            main_line1, = main_ax.plot(y_test)
            main_line2, = main_ax.plot(y_test*0.5)

            mse_line, = mse_ax.plot(np.linspace(0,1,60))


            plt.show()
        
        self.initialize_model_architecture(self.neurons_architecture)

        sess = tf.Session()
        self.saver.restore(sess,self.model_dir)


        #----------------TEST OUT PRELAUNCHED MODEL TO SEE IF ITS GOOD OR NOT----------------#
        if (mse_threshold is not None) and (type(mse_threshold) == float):
            test_mse = sess.run(self.cost, feed_dict = {self.x_holder: x_test, self.y_holder: y_test})
            #pred = self.predict(self.prediction_time,simple_mode=True)
            #test_mse = calculate_mse(pred.dropna().test,pred.dropna().prediction)
            if test_mse <= mse_threshold:
                if (save_model is True):
                    self.saver.save(sess,save_path = self.model_dir)
                sess.close()
                #gotta add all_mse for check later
                all_mse = []

            else:
                all_mse = []
                for e in range(epochs):
                    epoch_mse = []
                    if shuffle is False:
                        pass
                    else:
                        shuffled_index = np.random.permutation(np.arange(len(y_train)))
                        x_train = x_train[shuffled_index]
                        y_train = y_train[shuffled_index]
                    for i in range(0, len(y_train) // batch_size):
                        if shuffle is False:
                            start = np.random.choice(np.arange(len(y_train) - batch_size))
                        else:
                            start = i * batch_size
                        batch_x = x_train[start:start + batch_size]
                        batch_y = y_train[start:start + batch_size]
                        sess.run(self.opt, feed_dict={self.x_holder: batch_x, self.y_holder: batch_y})
                        if interactive == True:
                            if np.mod(i, 5) == 0:
                                pred = sess.run(self.architecture['out']['op'], feed_dict={self.x_holder: x_test})
                                main_line2.set_ydata(pred)
                                #plt.title(self.model_name+': Epoch ' + str(e) + ', Batch ' + str(i))
                                #plt.pause(0.01)
                                main_ax.set_title(self.model_name+': Epoch ' + str(e) + ', Batch ' + str(i))
                                plt.pause(0.01)
                        #test_mse = sess.run(self.cost, feed_dict = {self.x_holder: x_test, self.y_holder: y_test})
                        test_mse = sess.run(self.cost, feed_dict = {self.x_holder: x_test, self.y_holder: y_test})
                        all_mse.append(test_mse)
                        if (interactive == True):
                            if np.mod(i, 5) == 0:
                                mse_line.remove()
                                mse_ax.set_ylim([0,max(all_mse[-60:])])
                                mse_line, = mse_ax.plot(all_mse[-60:])
                                plt.pause(0.1)
                        if test_mse <= mse_threshold:
                            break
                        
                    # Objective 1: NEED TO FIND A WAY TO OPTIMIZE LEARNING CURVE
                    # Objective 2: plateauing mse

                    if (len(all_mse) == 0):
                        all_mse = [0]
                        break
                    elif all_mse[-1] <= mse_threshold:
                        break
                if interactive == True:
                    plt.close()
                if (save_model is True):
                    self.saver.save(sess,save_path = self.model_dir)
                sess.close()
                    
        #-----------------------------------------------------------------------------------#
        else:
            all_mse = []
            for e in range(epochs):
                shuffled_index = np.random.permutation(np.arange(len(y_train)))
                x_train = x_train[shuffled_index]
                y_train = y_train[shuffled_index]
                for i in range(0, len(y_train) // batch_size):
                    start = i * batch_size
                    batch_x = x_train[start:start + batch_size]
                    batch_y = y_train[start:start + batch_size]
                    sess.run(self.opt, feed_dict={self.x_holder: batch_x, self.y_holder: batch_y})
                    if interactive == True:
                        if np.mod(i, 5) == 0:
                            pred = sess.run(self.architecture['out']['op'], feed_dict={self.x_holder: x_test})
                            main_line2.set_ydata(pred)
                            main_ax.set_title(self.model_name+': Epoch ' + str(e) + ', Batch ' + str(i))
                            plt.pause(0.01)
                    test_mse = sess.run(self.cost, feed_dict = {self.x_holder: x_test, self.y_holder: y_test})
                    all_mse.append(test_mse)
                    if (interactive == True):
                        if np.mod(i, 5) == 0:
                            mse_line.remove()
                            mse_ax.set_ylim([0,max(all_mse[-60:])])
                            mse_line, = mse_ax.plot(all_mse[-60:])
                            plt.pause(0.01)
                    if test_mse <= mse_threshold:
                        break
                if (len(all_mse) == 0):
                    all_mse = [0]
                    break
                elif all_mse[-1] <= mse_threshold:
                    break
            if interactive == True:
                plt.close()
            if (save_model is True):
                self.saver.save(sess,save_path = self.model_dir)
            sess.close()
        if return_all_mse is True:
            return np.array(all_mse)


    def architecture_report(self):
        # extract weights and biases from our model
        with tf.Session() as sess:
            self.saver.restore(sess,self.model_dir)
            w1,w2,w3,w4,w_out = sess.run(['w1:0','w2:0','w3:0','w4:0','w_out:0'])
            b1,b2,b3,b4,b_out = sess.run(['b1:0','b2:0','b3:0','b4:0','b_out:0'])
        print('--------Architecture Report---------')
        print('Neurons Shapes:')
        print('w1:',str(w1.shape),'b1:',str(b1.shape))
        print('w2:',str(w2.shape),'b2:',str(b2.shape))
        print('w3:',str(w3.shape),'b3:',str(b3.shape))
        print('w4:',str(w4.shape),'b4:',str(b4.shape))
        return ([w1,w2,w3,w4,w_out],[b1,b2,b3,b4,b_out])

    def initialize_model_architecture(self,neurons_architecture,mode='trained'):
        tf.reset_default_graph()
        if ((mode == 'new') or (os.path.isfile(self.model_dir+'.meta') == False)):
            if ((mode == 'trained') and (os.path.isfile(self.model_dir+'.meta') == False)):
                print('No trained model found for neurons architecture: ', str(self.neurons_architecture))
                print('--Initializing New Model...')
            # If mode is new or there isn't an existing model
            #-----Establish Input and Test Data placeholders---#
            self.x_holder = tf.placeholder(dtype = tf.float32, shape = [None, self.input_size],name='x_holder')
            self.y_holder = tf.placeholder(dtype = tf.float32, shape = [None],name='y_holder')
            #----create hidden layers with desired architectural neurons----#
            self.architecture = {}
            for i in range(len(neurons_architecture)):
                self.architecture['layer'+str(i+1)] = {'weight':0,'bias':0,'op':0}
            for layer_index in range(len(neurons_architecture)):
                n_count = neurons_architecture[layer_index]
                # first layer has special indexes
                if layer_index == 0:
                    self.architecture['layer1']['weight'] = tf.Variable(tf.variance_scaling_initializer(mode='fan_avg',distribution='uniform',scale=1)([self.input_size,n_count]),name='w1')
                    self.architecture['layer1']['bias'] = tf.Variable(tf.zeros_initializer()([n_count]),name='b1')
                elif layer_index != 0:
                    # create activation ops in previous layer(s)
                    # if this is the second layer (we will be creating op for first layer - which is specifically different)
                    if layer_index == 1:
                        self.architecture['layer1']['op'] = tf.nn.relu(tf.add(tf.matmul(self.x_holder, self.architecture['layer1']['weight']), self.architecture['layer1']['bias']),name='hidden_1')
                    else:
                        # create activation op for the previous layer
                        self.architecture['layer'+str(layer_index)]['op'] = tf.nn.relu(tf.add(tf.matmul(self.architecture['layer'+str(layer_index - 1)]['op'], self.architecture['layer'+str(layer_index)]['weight']), self.architecture['layer'+str(layer_index)]['bias']),name='hidden_'+str(layer_index))
                    previous_n_count = neurons_architecture[layer_index - 1]
                    self.architecture['layer'+str(layer_index+1)]['weight'] = tf.Variable(tf.variance_scaling_initializer(mode='fan_avg',distribution='uniform',scale=1)([previous_n_count, n_count]),name='w'+str(layer_index+1))
                    self.architecture['layer'+str(layer_index+1)]['bias'] = tf.Variable(tf.zeros_initializer()([n_count]),name='b'+str(layer_index+1))
            #---create the last activation op after vars are established---#
            self.architecture['layer'+str(len(neurons_architecture))]['op'] = tf.nn.relu(tf.add(tf.matmul(self.architecture['layer'+str(len(neurons_architecture)-1)]['op'],self.architecture['layer'+str(len(neurons_architecture))]['weight']),self.architecture['layer'+str(len(neurons_architecture))]['bias'],name='hidden_'+str(len(neurons_architecture))))
            #---create the last output layer---#
            self.architecture['out'] = {'weight':0,'bias':0,'op':0}
            self.architecture['out']['weight'] = tf.Variable(tf.variance_scaling_initializer(mode='fan_avg',distribution='uniform',scale=1)([neurons_architecture[-1],1]),name='w_out')
            self.architecture['out']['bias'] = tf.Variable(tf.zeros_initializer()([1]),name='b_out')
            self.architecture['out']['op'] = tf.transpose(tf.add(tf.matmul(self.architecture['layer'+str(len(neurons_architecture))]['op'],self.architecture['out']['weight']), self.architecture['out']['bias']),name='out')

            #---create cost and optimizer functions
            self.cost = tf.reduce_mean(tf.squared_difference(self.architecture['out']['op'],self.y_holder),name='cost')
            self.opt = tf.train.AdamOptimizer().minimize(self.cost,name='optimizer')

        else:
            sess = tf.Session()

            #-----Import meta graph and extract trained variables values-----#
            self.saver = tf.train.import_meta_graph(self.model_dir+'.meta')
            self.saver.restore(sess,self.model_dir)
            graph = tf.get_default_graph()

            #----initialize placeholders----#
            self.x_holder = graph.get_tensor_by_name('x_holder:0')
            self.y_holder = graph.get_tensor_by_name('y_holder:0')

            #----initialize architecture---#
            self.architecture = {}

            #---searching for maximum hidden neuron layers---#
            for i in range(1,100):
                layer_key = 'layer'+str(i)
                self.architecture[layer_key] = {'weight':0,'bias':0,'op':0}
                finished = False
                for var in list(self.architecture[layer_key].keys()):
                    if var == 'op':
                        tensor_prefix = 'hidden_'
                    else:
                        tensor_prefix = var[0]
                    try:
                        self.architecture[layer_key][var] = graph.get_tensor_by_name(tensor_prefix+str(i)+':0')
                    except KeyError:
                        del self.architecture[layer_key]
                        finished = True
                        break
                if finished == True:
                    break
            #----load out layer into architecture---#
            self.architecture['out'] = {'weight':0,'bias':0,'op':0}
            self.architecture['out']['weight'] = graph.get_tensor_by_name('w_out:0')
            self.architecture['out']['bias'] = graph.get_tensor_by_name('b_out:0')
            self.architecture['out']['op'] = graph.get_tensor_by_name('out:0')

            #----cost tensor and optimizer functions---#
            self.cost = graph.get_tensor_by_name('cost:0')
            self.opt = graph.get_operation_by_name('optimizer')

            sess.close()

        #----Finish initialization---#
        self.saver = tf.train.Saver()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        self.saver.save(sess,save_path=self.model_dir)
        sess.close()


    def set_hyper_params(self,batch_size=None,epochs=None,test_size=None):
        if (batch_size is not None) and (type(batch_size) == int):
            self.hyper_params['batch_size'] = batch_size
        if (epochs is not None) and (type(epochs) == int):
            self.hyper_params['epochs'] = epochs
        if test_size is not None:
            self.hyper_params['test_size'] = test_size

    def predict(self,prediction_time,simple_mode=False):
        #----PREDICT USING AMOUNT OF DATA SPECIFIED FOR TEST_SIZE----#
        df = pd.DataFrame().append(self.dataframe.tail(self.hyper_params['test_size']))
        df[self.prediction_target+'_pred'] = df[self.prediction_target].shift(-prediction_time)
        #----FIT SCALER TO PREDICTION DATAFRAME WITH SHIFTED DATA----#
        df_to_fit = pd.DataFrame().append(self.dataframe)
        df_to_fit[self.prediction_target+'_pred'] = df_to_fit[self.prediction_target].shift(-prediction_time)
        df_to_fit = df_to_fit.dropna()
        #---FITTING THE LATEST AMOUNT OF DATA FOR PREDICTION TIME
        #---[this means that we have no testing data yet, aka we're using out best trained model to predict]
        self.scaler.fit(df_to_fit,y=df_to_fit[self.prediction_target+'_pred'])

        if simple_mode is False:
            dt_stamps = pd.to_datetime(df.index)
            latest_stamp = dt_stamps[-1] 

            # Now we know the start and end time, we'll propagate the time forward with delta as interval        
            pred_stamps = dt_stamps


            # Setting up necessary variables to identify prediction stamp(s)
            end_time = None
            start_time = None
            next_day_delta = None
            next_week_delta = None
            df_interval = None

            all_dfs = []
            for group in df.groupby(pd.to_datetime(df.index).date):
                all_dfs.append(group[1])
            for i in range(len(all_dfs)-1):
                day_1 = all_dfs[i]
                day_2 = all_dfs[i+1]
                day_1.index = pd.to_datetime(day_1.index)
                day_2.index = pd.to_datetime(day_2.index)
                if next_week_delta is None:
                    if ((day_1.index[-1].dayofweek == 4) and (day_2.index[0].dayofweek == 0)):
                            next_week_delta = day_2.index[0] - day_1.index[-1]
                if next_day_delta is None:
                    if (day_1.index[-1].dayofweek != 4):
                        next_day_delta = day_2.index[0] - day_1.index[-1]
                
                if df_interval is None:
                    if ((len(day_1) == 1) and (len(day_2) == 1)):
                        df_interval = 'daily'
                    else:
                        df_interval = 'intraday'
                
                if (df_interval == 'intraday') and ((end_time is None) or (start_time is None)):
                    start_time = day_2.index[0].time()
                    end_time = day_2.index[-1].time()
                else:
                    break
            #------------------------------#

            for t in range(prediction_time):
                # if we're predicting on a daily dataframe or above
                if df_interval != 'intraday':
                    if pred_stamps[-1].dayofweek == 0: # if we're on a monday
                        delta = pred_stamps[-2] - pred_stamps[-3]
                    else:
                        delta = pred_stamps[-1] - pred_stamps[-2]
                else:
                    # if latest time is already at end time
                    if pred_stamps[-1].time() == end_time:
                        # if we are on a friday, we gotta add the next monday
                        if pred_stamps[-1].dayofweek == 4:
                            delta = next_week_delta                 
                        else:
                            delta = next_day_delta
                    # if latest time is already at start time
                    elif pred_stamps[-1].time() == start_time:
                        delta = pred_stamps[-2] - pred_stamps[-3]
                    # else
                    else:
                        delta = dt_stamps[-1] - dt_stamps[-2]
                pred_stamps = pred_stamps.append(pd.Index([pred_stamps[-1] + delta]))
                pred_stamps = pred_stamps[1:]
            #If we're predicting on a daily dataframe
            if df_interval != 'intraday':
                prediction_indexes = pd.Index([str(dt).split(' ')[0] for dt in pred_stamps])
                test_indexes = pd.Index([str(dt).split(' ')[0] for dt in dt_stamps])
            else:
                prediction_indexes = pd.Index([str(dt) for dt in pred_stamps])
                test_indexes = pd.Index([str(dt) for dt in dt_stamps])
        else:
            pass
        #---AFTER WE FITTED OUR LATEST "FITTABLE" AMOUNT OF DATA FOR "TRAINING",
        #---WE WILL USE IT TO TRANSFORM OUR FULL DATATAFRAME
        target_index = df.columns.get_loc(self.prediction_target+'_pred')
        scaled_data = self.scaler.transform(df)

        test_scaled = scaled_data[:,target_index]
        params_scaled = np.delete(scaled_data,target_index,1)

        sess = tf.Session()
        self.saver.restore(sess,self.model_dir)
        try:
            pred_scaled = sess.run(self.architecture['out']['op'],feed_dict={self.x_holder: params_scaled})
            sess.close()
        except TypeError:
            sess.close()
            self.initialize_model_architecture(self.neurons_architecture)
            sess = tf.Session()
            self.saver.restore(sess,self.model_dir)
            pred_scaled = sess.run(self.architecture['out']['op'],feed_dict={self.x_holder: params_scaled})
            sess.close()

        

        to_unscale = np.insert(params_scaled,target_index,pred_scaled[0,:],axis=1)
        
        test_unscaled = df[self.prediction_target].values
        pred_unscaled = self.scaler.inverse_transform(to_unscale)[:,target_index]

        if simple_mode is False:
            unscaled_pred = pd.Series(data = pred_unscaled,index = prediction_indexes)
            unscaled_test = pd.Series(data = test_unscaled,index = test_indexes)

            scaled_pred = pd.Series(data = pred_scaled[0,:], index = prediction_indexes)
            scaled_test = pd.Series(data = test_scaled, index = test_indexes)
        else:
            unscaled_pred = pd.Series(data=pred_unscaled)
            unscaled_test = pd.Series(data=test_unscaled[prediction_time:])

            scaled_pred = pd.Series(data = pred_scaled[0,:])
            scaled_test = pd.Series(data = test_scaled[prediction_time:])


        scaled_result = {'test': scaled_test,
                        'prediction': scaled_pred}

        unscaled_result = {'test': unscaled_test,
                        'prediction': unscaled_pred}

        #result['unscaled_pred'] = unscaled_pred
        #result['unscaled_test'] = unscaled_test

        return {'scaled': pd.DataFrame(scaled_result),
                'unscaled': pd.DataFrame(unscaled_result)}
        

    def extract_train_df(self,scale=True):

        df = pd.DataFrame().append(self.dataframe)

        df[self.prediction_target+'_pred'] = df[self.prediction_target].shift(-self.prediction_time)
        df = df.dropna()
        target_index = df.columns.get_loc(self.prediction_target+'_pred')

        split = self.hyper_params['test_size']
        if scale is True:
            df = self.scaler.fit_transform(df,y=df[self.prediction_target+'_pred'])
            train = df[:-split,:]
            test = df[-split:,:]
        else:
            train = df.values[:-split,:]
            test = df.values[-split:,:]

        #train = self.scaler.fit_transform(train,y=train[self.prediction_target+'_pred'])
        #test = self.scaler.fit_transform(test,y=test[self.prediction_target+'_pred'])

        y_train = train[:,target_index]
        x_train = np.delete(train,target_index,1)
        y_test = test[:,target_index]
        x_test = np.delete(test,target_index,1)

        return x_train,y_train,x_test,y_test