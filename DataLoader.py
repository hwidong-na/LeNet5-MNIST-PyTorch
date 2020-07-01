#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import numpy
import random
import pdb
import os
import threading
import time
import math
from queue import Queue

def loadIMG(img, num_aug=1):
    if num_aug > 1:
        feat = []
        for i in range(num_aug):
            perm = numpy.random.permutation(img.shape[0])
            feat.append(img[perm])
        return torch.stack(feat)
    return img


def round_down(num, divisor):
    return num - (num%divisor)

class DataLoader(object):
    def __init__(self, dataset, batch_size, max_spl_per_cls, nDataLoaderThread, gSize, maxQueueSize = 10, **kwargs):
        self.dataset = dataset;
        self.nWorkers = nDataLoaderThread;
        self.max_spl_per_cls = max_spl_per_cls;
        self.batch_size = batch_size;
        self.maxQueueSize = maxQueueSize;

        self.data_dict = {};
        self.data_list = [];
        self.nFiles = 0;
        self.gSize  = gSize; ## number of clips per sample (e.g. 1 for softmax, 2 for triplet or pm)

        self.dataLoaders = [];
        
        for data, label in dataset:
            if not (label in self.data_dict):
                self.data_dict[label] = [];

            self.data_dict[label].append(data);

        # print("Total # classes: ", len(self.data_dict))
        # print("Total # samples: ", sum(map(len, self.data_dict.values())))
        
        self.datasetQueue = Queue(self.maxQueueSize);

        self.batch_size = min(len(self.data_dict), self.batch_size)
    

    def dataLoaderThread(self, nThreadIndex):
        
        index = nThreadIndex*self.batch_size;

        if(index >= self.nFiles):
            return;

        while(True):
            if(self.datasetQueue.full() == True):
                time.sleep(1.0);
                continue;

            if(index+self.batch_size > self.nFiles): # drop last
                break;

            in_data = [];
            for ii in range(0,self.gSize):
                feat = []
                for ij in range(index,index+self.batch_size):
                    feat.append(loadIMG(self.data_list[ij][ii]));
                in_data.append(torch.stack(feat, dim=0));

            in_data = torch.stack(in_data, axis=1) #(batch,seg,c,h,w)
            in_label = numpy.asarray(self.data_label[index:index+self.batch_size]);
            
            self.datasetQueue.put([in_data, in_label]);

            index += self.batch_size*self.nWorkers;

    def __iter__(self):

        dictkeys = list(self.data_dict.keys());
        dictkeys.sort()

        lol = lambda lst, sz: [lst[i:i+sz] for i in range(0, len(lst), sz)]

        flattened_list = []
        flattened_label = []

        ## Data for each class
        for findex, key in enumerate(dictkeys):
            data    = self.data_dict[key]
            numSeg  = round_down(min(len(data),self.max_spl_per_cls),self.gSize)
            
            rp      = lol(numpy.random.permutation(len(data))[:numSeg],self.gSize)
            flattened_label.extend([findex] * (len(rp)))
            for indices in rp:
                flattened_list.append([data[i] for i in indices])

        ## Data in random order
        mixid           = numpy.random.permutation(len(flattened_label))
        mixlabel        = []
        mixmap          = []

        ## Prevent two pairs of the same classes in the same batch
        for ii in mixid:
            startbatch = len(mixlabel) - len(mixlabel) % self.batch_size
            if flattened_label[ii] not in mixlabel[startbatch:]:
                mixlabel.append(flattened_label[ii])
                mixmap.append(ii)

        self.data_list  = [flattened_list[i] for i in mixmap]
        self.data_label = [flattened_label[i] for i in mixmap]
        
        ## Iteration size
        self.nFiles = len(self.data_label);
        # print("Total # batches: ", self.nFiles)

        ### Make and Execute Threads...
        for index in range(0, self.nWorkers):
            self.dataLoaders.append(threading.Thread(target = self.dataLoaderThread, args = [index]));
            self.dataLoaders[-1].start();

        return self;


    def __next__(self):

        while(True):
            isFinished = True;
            
            if(self.datasetQueue.empty() == False):
                return self.datasetQueue.get();
            for index in range(0, self.nWorkers):
                if(self.dataLoaders[index].is_alive() == True):
                    isFinished = False;
                    break;

            if(isFinished == False):
                time.sleep(1.0);
                continue;


            for index in range(0, self.nWorkers):
                self.dataLoaders[index].join();

            self.dataLoaders = [];
            raise StopIteration;


    def __call__(self):
        pass;

    def qsize(self):
        return self.datasetQueue.qsize();



class UnlabeledDataLoader(DataLoader):

    def __iter__(self):

        dictkeys = list(self.data_dict.keys());
        dictkeys.sort()

        lol = lambda lst, sz: [lst[i:i+sz] for i in range(0, len(lst), sz)]

        flattened_list = []
        flattened_label = []

        for findex, key in enumerate(dictkeys):
            assert len(self.data_dict[key]) == 1
            flattened_list.append(self.data_dict[key])
            flattened_label.append(findex)

        ## Data in random order
        mixid           = numpy.random.permutation(len(flattened_label))
        mixlabel        = []
        mixmap          = []

        ## Prevent two pairs of the same speaker in the same batch
        for ii in mixid:
            startbatch = len(mixlabel) - len(mixlabel) % self.batch_size
            if flattened_label[ii] not in mixlabel[startbatch:]:
                mixlabel.append(flattened_label[ii])
                mixmap.append(ii)

        self.data_list  = [flattened_list[i] for i in mixmap]
        self.data_label = [flattened_label[i] for i in mixmap]
        
        ## Iteration size
        self.nFiles = len(self.data_label);

        ### Make and Execute Threads...
        for index in range(0, self.nWorkers):
            self.dataLoaders.append(threading.Thread(target = self.dataLoaderThread, args = [index]));
            self.dataLoaders[-1].start();

        return self;

    def dataLoaderThread(self, nThreadIndex):
        
        index = nThreadIndex*self.batch_size;

        if(index >= self.nFiles):
            return;

        while(True):
            if(self.datasetQueue.full() == True):
                time.sleep(1.0);
                continue;

            if(index+self.batch_size > self.nFiles): # drop last
                break;

            feat = []
            for ij in range(index,index+self.batch_size):
                # load augmentaed image
                feat.append(loadIMG(self.data_list[ij][0], num_aug=self.gSize));

            in_data = torch.stack(feat, axis=0); #(batch,seg,max_audio)

            in_label = numpy.asarray(self.data_label[index:index+self.batch_size])
            
            self.datasetQueue.put([in_data, in_label]);

            index += self.batch_size*self.nWorkers;
