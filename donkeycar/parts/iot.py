#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 12:32:53 2017

@author: wroscoe
"""
import os
import sys
import time
import json
import datetime
import random
import glob
import numpy as np
import pandas as pd

from PIL import Image
from donkeycar import utils

from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient


class Tub(object):
    """
    A datastore to store sensor data in a key, value format.

    Accepts str, int, float, image_array, image, and array data types.

    For example:

    #Create a tub to store speed values.
    >>> path = '~/mydonkey/test_tub'
    >>> inputs = ['user/speed', 'cam/image']
    >>> types = ['float', 'image']
    >>> t=Tub(path=path, inputs=inputs, types=types)

    """

    def __init__(self, path, inputs=None, types=None):

        self.path = os.path.expanduser(path)
        print('path_in_tub:', self.path)
        self.meta_path = os.path.join(self.path, 'meta.json')
        self.df = None

        exists = os.path.exists(self.path)

        if exists:
            #load log and meta
            print("Tub exists: {}".format(self.path))
            with open(self.meta_path, 'r') as f:
                self.meta = json.load(f)
            self.current_ix = self.get_last_ix() + 1

        elif not exists and inputs:
            print('Tub does NOT exist. Creating new tub...')
            #create log and save meta
            os.makedirs(self.path)
            self.meta = {'inputs': inputs, 'types': types}
            with open(self.meta_path, 'w') as f:
                json.dump(self.meta, f)
            self.current_ix = 0
            print('New tub created at: {}'.format(self.path))
        else:
            msg = "The tub path you provided doesn't exist and you didnt pass any meta info (inputs & types)" + \
                  "to create a new tub. Please check your tub path or provide meta info to create a new tub."

            raise AttributeError(msg)

        self.start_time = time.time()


    def get_last_ix(self):
        index = self.get_index()
        return max(index)

    def update_df(self):
        df = pd.DataFrame([self.get_json_record(i) for i in self.get_index(shuffled=False)])
        self.df = df

    def get_df(self):
        if self.df is None:
            self.update_df()
        return self.df


    def get_index(self, shuffled=True):
        files = next(os.walk(self.path))[2]
        record_files = [f for f in files if f[:6]=='record']
        
        def get_file_ix(file_name):
            try:
                name = file_name.split('.')[0]
                num = int(name.split('_')[1])
            except:
                num = 0
            return num

        nums = [get_file_ix(f) for f in record_files]
        
        if shuffled:
            random.shuffle(nums)
        else:
            nums = sorted(nums)
            
        return nums 


    @property
    def inputs(self):
        return list(self.meta['inputs'])

    @property
    def types(self):
        return list(self.meta['types'])

    def get_input_type(self, key):
        input_types = dict(zip(self.inputs, self.types))
        return input_types.get(key)

    def write_json_record(self, json_data):
        path = self.get_json_record_path(self.current_ix)
        try:
            with open(path, 'w') as fp:
                json.dump(json_data, fp)
                #print('wrote record:', json_data)
        except TypeError:
            print('troubles with record:', json_data)
        except FileNotFoundError:
            raise
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise

    def get_num_records(self):
        import glob
        files = glob.glob(os.path.join(self.path, 'record_*.json'))
        return len(files)




    def make_record_paths_absolute(self, record_dict):
        #make paths absolute
        d = {}
        for k, v in record_dict.items():
            if type(v) == str: #filename
                if '.' in v:
                    v = os.path.join(self.path, v)
            d[k] = v

        return d




    def check(self, fix=False):
        '''
        Iterate over all records and make sure we can load them.
        Optionally remove records that cause a problem.
        '''
        print('Checking tub:%s.' % self.path)
        print('Found: %d records.' % self.get_num_records())
        problems = False
        for ix in self.get_index(shuffled=False):
            try:
                self.get_record(ix)
            except:
                problems = True
                if fix == False:
                    print('problems with record:', self.path, ix)
                else:
                    print('problems with record, removing:', self.path, ix)
                    self.remove_record(ix)
        if not problems:
            print("No problems found.")

    def remove_record(self, ix):
        '''
        remove data associate with a record
        '''
        record = self.get_json_record_path(ix)
        os.unlink(record)

    def put_record(self, data):
        """
        Save values like images that can't be saved in the csv log and
        return a record with references to the saved values that can
        be saved in a csv.
        """
        json_data = {}
        self.current_ix += 1
        
        for key, val in data.items():
            typ = self.get_input_type(key)

            if typ in ['str', 'float', 'int', 'boolean']:
                json_data[key] = val

            elif typ is 'image':
                path = self.make_file_path(key)
                val.save(path)
                json_data[key]=path

            elif typ == 'image_array':
                img = Image.fromarray(np.uint8(val))
                name = self.make_file_name(key, ext='.jpg')
                img.save(os.path.join(self.path, name))
                json_data[key]=name

            else:
                msg = 'Tub does not know what to do with this type {}'.format(typ)
                raise TypeError(msg)

        self.write_json_record(json_data)
        return self.current_ix


    def get_json_record_path(self, ix):
        return os.path.join(self.path, 'record_'+str(ix)+'.json')

    def get_json_record(self, ix):
        path = self.get_json_record_path(ix)
        try:
            with open(path, 'r') as fp:
                json_data = json.load(fp)
        except UnicodeDecodeError:
            raise Exception('bad record: %d. You may want to run `python manage.py check --fix`' % ix)
        except FileNotFoundError:
            raise
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise

        record_dict = self.make_record_paths_absolute(json_data)
        return record_dict


    def get_record(self, ix):

        json_data = self.get_json_record(ix)
        data = self.read_record(json_data)
        return data



    def read_record(self, record_dict):
        data={}
        for key, val in record_dict.items():
            typ = self.get_input_type(key)

            #load objects that were saved as separate files
            if typ == 'image_array':
                img = Image.open((val))
                val = np.array(img)

            data[key] = val


        return data


    def make_file_name(self, key, ext='.png'):
        name = '_'.join([str(self.current_ix), key, ext])
        name = name = name.replace('/', '-')
        return name

    def delete(self):
        """ Delete the folder and files for this tub. """
        import shutil
        shutil.rmtree(self.path)

    def shutdown(self):
        pass


    def get_record_gen(self, record_transform=None, shuffle=True, df=None):

        if df is None:
            df = self.get_df()


        while True:
            for row in self.df.iterrows():
                if shuffle:
                    record_dict = df.sample(n=1).to_dict(orient='record')[0]

                if record_transform:
                    record_dict = record_transform(record_dict)

                record_dict = self.read_record(record_dict)

                yield record_dict


    def get_batch_gen(self, keys, record_transform=None, batch_size=128, shuffle=True, df=None):

        record_gen = self.get_record_gen(record_transform, shuffle=shuffle, df=df)

        if keys == None:
            keys = list(self.df.columns)

        while True:
            record_list = []
            for _ in range(batch_size):
                record_list.append(next(record_gen))

            batch_arrays = {}
            for i, k in enumerate(keys):
                arr = np.array([r[k] for r in record_list])
                # if len(arr.shape) == 1:
                #    arr = arr.reshape(arr.shape + (1,))
                batch_arrays[k] = arr

            yield batch_arrays


    def get_train_gen(self, X_keys, Y_keys, batch_size=128, record_transform=None, df=None):

        batch_gen = self.get_batch_gen(X_keys + Y_keys,
                                       batch_size=batch_size, record_transform=record_transform, df=df)

        while True:
            batch = next(batch_gen)
            X = [batch[k] for k in X_keys]
            Y = [batch[k] for k in Y_keys]
            yield X, Y


    def get_train_val_gen(self, X_keys, Y_keys, batch_size=128, record_transform=None, train_frac=.8):
        train_df = train=self.df.sample(frac=train_frac,random_state=200)
        val_df = self.df.drop(train_df.index)

        train_gen = self.get_train_gen(X_keys=X_keys, Y_keys=Y_keys, batch_size=batch_size,
                                       record_transform=record_transform, df=train_df)

        val_gen = self.get_train_gen(X_keys=X_keys, Y_keys=Y_keys, batch_size=batch_size,
                                       record_transform=record_transform, df=val_df)

        return train_gen, val_gen





class TubWriter(Tub):
    def __init__(self, *args, **kwargs):
        super(TubWriter, self).__init__(*args, **kwargs)

    def run(self, *args):
        '''
        API function needed to use as a Donkey part.

        Accepts values, pairs them with their inputs keys and saves them
        to disk.
        '''
        assert len(self.inputs) == len(args)

        self.record_time = int(time.time() - self.start_time)
        record = dict(zip(self.inputs, args))
        self.put_record(record)

def customCallback(client, userdata, message):
    print("Received a new message")
    print(message.payload)
    print("From Topic:")
    print(message.topic)
    print("--------------------\n\n")

class IotPush(Tub):
    def __init__(self, inputs=None, types=None):
        self.client = self.connect()
        self.meta = {'inputs': inputs, 'types': types}
        self.start_time = time.time()
        self.current_ix = 1

    def update(self):
        time.sleep(1)

    def put_record(self, data):
        """
        Save values like images that can't be saved in the csv log and
        return a record with references to the saved values that can
        be saved in a csv.
        """
        json_data = {}
        self.current_ix += 1

        for key, val in data.items():
            typ = self.get_input_type(key)

            if typ in ['str', 'float', 'int', 'boolean']:
                json_data[key] = val

            elif typ is 'image':
                pass

            elif typ == 'image_array':
                pass

            else:
                msg = 'Tub does not know what to do with this type {}'.format(typ)
                raise TypeError(msg)

        self.write_json_record(json_data)
        return self.current_ix
	

    def connect(self):

        base_path = "/opt/devicesdk"
        rootCAPath = os.path.join(base_path, "root-CA.crt")
        privateKeyPath = os.path.join(base_path, "donkeyng.private.key")
        certificatePath = os.path.join(base_path, "donkeyng.cert.pem")
        topic = "AutonomousVehicles"
        clientId = "BasicPubSub"
        host  = "a3u210mv68xw24.iot.us-east-1.amazonaws.com"

        #print (rootCAPath, privateKeyPath, certificatePath)

        # Init AWSIoTMQTTClient
        useWebsocket = False
        myAWSIoTMQTTClient = None
        if useWebsocket:
            myAWSIoTMQTTClient = AWSIoTMQTTClient(clientId, useWebsocket=True)
            myAWSIoTMQTTClient.configureEndpoint(host, 443)
            myAWSIoTMQTTClient.configureCredentials(rootCAPath)
        else:
            myAWSIoTMQTTClient = AWSIoTMQTTClient(clientId)
            myAWSIoTMQTTClient.configureEndpoint(host, 8883)
            myAWSIoTMQTTClient.configureCredentials(rootCAPath, privateKeyPath, certificatePath)

        # AWSIoTMQTTClient connection configuration
        myAWSIoTMQTTClient.configureAutoReconnectBackoffTime(1, 32, 20)
        myAWSIoTMQTTClient.configureOfflinePublishQueueing(-1)  # Infinite offline Publ$
        myAWSIoTMQTTClient.configureDrainingFrequency(2)  # Draining: 2 Hz
        myAWSIoTMQTTClient.configureConnectDisconnectTimeout(10)  # 10 sec
        myAWSIoTMQTTClient.configureMQTTOperationTimeout(5)  # 5 sec

        # Connect and subscribe to AWS IoT
        myAWSIoTMQTTClient.connect()
        myAWSIoTMQTTClient.subscribe(topic, 1, customCallback)
        time.sleep(2)

        return myAWSIoTMQTTClient

    def run_threaded(self, *args):
        assert len(self.inputs) == len(args)
        
        self.record_time = int(time.time() - self.start_time)
        record = dict(zip(self.inputs, args))
        self.put_record(record)
        #time.sleep(1)
    
    def write_json_record(self, json_data):
        topic = "AutonomousVehicles"
        self.client.publish(topic, json.dumps(json_data), 1)

if __name__=="__main__":
    
    inputs = [10, 20, "A string"]
    types = ['int', 'int', 'str']
    iot = IotPush(inputs=inputs, types=types)
    topic = "AutonomousVehicles"
    for i in range (0,10):
        iot.run_threaded()
        time.sleep(1)
