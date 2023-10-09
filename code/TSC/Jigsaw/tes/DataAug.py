#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 14:49:20 2023

@author: mariohabibfathi
"""

import itertools
import random
import numpy as np

class DataAugmentation:
    """
    A class representing the adaptation for the Jigsaw technique that is used for image unspervised 
    learning to be used for time series classification taken from
    
    
    Attributes:
        segments_num (int): The total number of segments.
        permutations_list (list): The generated permutation list used for adaptation.

    Reference:
    @inproceedings{noroozi2016unsupervised,
      title={Unsupervised learning of visual representations by solving jigsaw puzzles},
      author={Noroozi, Mehdi and Favaro, Paolo},
      booktitle={European conference on computer vision},
      pages={69--84},
      year={2016},
      organization={Springer}
    }

    Methods:
    - __init__(self, segments_num=3, gap=0): Constructor method.
    - JigSaw(self, Dataset, Augmented_data=True, generat_gap=True, num_of_new_copies=None): Jigsaw data augmentation method.
    - Devide_ser(self, ser): Divide the series into segments.
    - merge(self, ser): Merge segments.
    - _generat_permutations_list(self): Generate permutations list.
    - _genliswithcond4(self): Generate permutations with condition.
    - _check_element(self, lis): Check the condition for elements.
    - _check_list(self, lis): Check the condition for the list.
    - Permutation(self, ser, generat_gap=True): Perform permutations on data.
    - Generate_gap(self, ser): Generate gaps in data.
    - _generat_gap_list(self): Generate gap list.
    - _generat_series(self, ser): Generate augmented series.
    - reshape_data(self, dataset): Reshape data.
    - generat_data(self, ser): Generate data.
    
    """
    def __init__(self,segments_num = 3,gap=0):
        self.segments_num = segments_num
        # self.permutations_list= list(itertools.permutations(self._generat_permutations_list()))
        # self.permutations_list.pop(0)
        # random.shuffle(self.permutations_list)
        self._generat_permutations_list()
        # print(self.permutations_list)

    def JigSaw(self,Dataset,Augmented_data = True,generat_gap = True, num_of_new_copies = None):
        """
        

        Parameters
        ----------
        Dataset : TYPE
            DESCRIPTION.
        Augmented_data : TYPE, optional
            DESCRIPTION. The default is True.
        generat_gap : TYPE, optional
            DESCRIPTION. The default is True.
        num_of_new_copies : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        Augmented_dataset : TYPE
            DESCRIPTION.
        New_Data : TYPE
            DESCRIPTION.

        """
        New_Data = []
        Augmented_dataset = []
        
        if num_of_new_copies is None:
            self.num_of_new_copies = self.segments_num
        else:
            self.num_of_new_copies = num_of_new_copies
        if Augmented_data:
            self.Augmented_data = Augmented_data
            Augmented_dataset = self._generat_series(Dataset)
            for i in range(0,len(Augmented_dataset),self.num_of_new_copies):
                dev = self.Devide_ser(Augmented_dataset[i:i+self.num_of_new_copies])
                New_Data.append(self.Permutation(dev,generat_gap))
            # return Augmented_dataset, New_Data
            
        else:
            # print("here")
            self.Augmented_data = False
            for x in Dataset:
                dev = self.Devide_ser(x)
                New_Data.append(self.Permutation(dev,generat_gap))
            Augmented_dataset = Dataset
        New_Data = self.reshape_data(New_Data)
        return Augmented_dataset,New_Data


    def Devide_ser(self,ser):
        
        
        new_ser = []
        if self.Augmented_data:
            self.divsion_step = int(len(ser[0])/self.segments_num)
            if self.divsion_step == 0:
                self.divsion_step = 1
            star = 0
            divsion_point = self.divsion_step
            for s in ser:
                new_copy_ser = []
                for _ in range(self.segments_num):
                    if _ == self.segments_num -1:
                        new_copy_ser.append(s)
                    else:
                        new_copy_ser.append(s[star:divsion_point])
                        s = s[divsion_point:]
                new_ser.append(new_copy_ser)
            return new_ser

        else:            
            self.divsion_step = int(len(ser)/self.segments_num)
            # print(self.divsion_step)
            if self.divsion_step == 0:
                self.divsion_step = 1
            star = 0
            divsion_point = self.divsion_step
            for _ in range(self.segments_num):
                if _ == self.segments_num -1:
                    # new_ser.append(ser[star:divsion_point])
                    new_ser.append(ser)
                    # print("sta :",star)
                    # print("end :",divsion_point)
                    # print("sss")
                else:
                # print(ser)
                # print(star)
                # print(divsion_point)
                    new_ser.append(ser[star:divsion_point])
                    ser = ser[divsion_point:]
                    # del ser[star:divsion_point]
                    # print("sta :",star)
                    # print("end :",divsion_point)
                    # print("sss")
                    # star +=divsion_step+1
                    # divsion_point+=divsion_step+1
            return new_ser
    
    def merge(self,ser):
        
        return [item for sublist in ser for item in sublist]
    
    # def _generat_permutations_list(self):
        
    #     lis = []
    #     # if self.segments_num >= 9:
    #     #     self.segments_num = 9
    #     # if self.segments_num < 2:
    #     #     self.segments_num = 2
    #     for i in range(self.segments_num):
    #         lis.append(i)
    #     # print(lis)
    #     return lis
    
    # def _generat_permutations_list(self):
        
    #     self.permutations_list = []
    #     if self.segments_num < 2:
    #         self.segments_num = 2
    #     lis = list(range(0, self.segments_num))
        
    #     if self.segments_num < 4:
    #         self.permutations_list= list(itertools.permutations(lis))
    #         self.permutations_list.pop(0)
    #         random.shuffle(self.permutations_list)        
        
        
    #     else:
    #         num_permutations = self.segments_num * 5
            
    #         # for i in range(num_permutations):
    #         #     permutation = random.sample(lis, len(lis))
    #         #     if permutation not in self.permutations_list:
    #         #         self.permutations_list.append(permutation)
    #         #     else:
    #         #         print('here before ',i)
    #         #         i = i-1
    #         #         print('here ',i)
    #         #     random.shuffle(self.permutations_list)
    #         i = 0
    #         while(i<num_permutations):
    #             permutation = random.sample(lis, len(lis))
    #             if self._check_list(permutation) and permutation not in self.permutations_list and permutation != lis:        
    #                 self.permutations_list.append(permutation)
    #                 i+=1
    #             else:
    #                 print('here ',i,'perm ',permutation)
    #             random.shuffle(self.permutations_list)
        # print(self.segments_num)
        # print(len(self.permutations_list))
        # print(self.permutations_list)

    def _generat_permutations_list(self):
        
        self.permutations_list = []
        if self.segments_num < 2:
            self.segments_num = 2
        lis = list(range(0, self.segments_num))
        
        if self.segments_num < 4:
            self.permutations_list= list(itertools.permutations(lis))
            self.permutations_list.pop(0)
            random.shuffle(self.permutations_list)        
        
        
        else:
            num_permutations = self.segments_num * 5
            i = 0
            while(i<num_permutations):
                if self.segments_num < 6:

                    permutation = random.sample(lis, len(lis))
                    if self._check_list(permutation) and permutation not in self.permutations_list and permutation != lis:        
                        self.permutations_list.append(permutation)
                        i+=1
                    else:
                        print('here ',i,'perm ',permutation)
                    random.shuffle(self.permutations_list)
                else:
                    permutation = self._genliswithcond4()
                    if self._check_list(permutation) and permutation not in self.permutations_list and permutation != lis:        
                        self.permutations_list.append(permutation)
                        i+=1
                    else:
                        print('here ',i)
                    random.shuffle(self.permutations_list)






    def _genliswithcond4(self)    :
        lis_genrated = [random.randint(0, 3)]
        all_elements =  list(range(0,self.segments_num))
        all_elements.remove(lis_genrated[0])
        k = 1 
        while (k <self.segments_num ):
            if k <= self.segments_num -2:
                candidate_num = random.randint(max(0, lis_genrated[k-1]-4), min(self.segments_num-1, lis_genrated[k-1]+4))
                if candidate_num not in lis_genrated :
                    lis_genrated.append(candidate_num)
                    if  min(all_elements) not in lis_genrated and (abs(min(all_elements)-k)>=3) :
                        lis_genrated[-1] = min(all_elements)   
                        candidate_num = min(all_elements)
                    all_elements.remove(candidate_num)
                    k+=1  
                check = self._check_element(lis_genrated)
                if check != True:
                    lis_genrated.remove(check)
                    all_elements.append(check)
                    k-=1
            else:
                lis_genrated.append(all_elements[0])
                k+=1            
    
        return lis_genrated
    
    
    
    def _check_element(self,lis):  
        n = len(lis)
        indices = list(range(n))
        
        for i in indices:
            if abs(i - lis[i])>=4:
                return lis[i]
        return True

    def _check_list(self,lis):  
        n = len(lis)
        indices = list(range(n))
        
        for i in indices:
            if abs(i - lis[i])>=4:
                return False
        return True
    
    def Permutation(self,ser,generat_gap = True):
        
        new_ser = []
        if self.Augmented_data:
            
            
            # print(self.segments_num)
            # print(self.permutations_list)
            # if len(self.permutations_list)<2:
            #     sample = 1
            # else:
                
            if len(self.permutations_list)< 2:
                sample = 1
            elif self.segments_num>=self.num_of_new_copies:
                sample = self.segments_num
            elif len(self.permutations_list)<self.num_of_new_copies:
            #     print('here')
                sample = len(self.permutations_list)
            else:
                sample = self.num_of_new_copies
                
                
            # print("Len Perm list ",len(self.permutations_list))
            # print("Perm list ",self.permutations_list)
            # print("sample ",sample)
            
            perm = random.sample(self.permutations_list,sample)
            # print("Perm ",perm)
            # print("Len Perm ",len(perm))
            # print("Len ser ",len(ser))
            for s in range (len(ser)):
                new_copy_ser = []
                # print(s)
                # print(perm)
                # print(len(ser))
                # print("s",s)
                # print(perm[s])
                
                permutation_index = s%len(perm)
                for i in perm[permutation_index]:
                    new_copy_ser.append(ser[s][i])
                # print(len(new_ser))
                if generat_gap == True:
                    self._generat_gap_list()
                    for i in range(len(new_copy_ser)):
                        # print(x)
                        new_copy_ser[i] = self.Generate_gap(new_copy_ser[i])
                        # print(x)
                # print(self.permutations_list)
                # print(random.choice(self.permutations_list))
                new_copy_ser = self.merge(new_copy_ser)
                new_ser.append(new_copy_ser)
            return new_ser , perm
        else:
            
            perm = random.choice(self.permutations_list)
            # print("Perm ",perm)
            for i in perm:
                new_ser.append(ser[i])
            # print(len(new_ser))
            if generat_gap == True:
                self._generat_gap_list()
                for i in range(len(new_ser)):
                    # print(x)
                    new_ser[i] = self.Generate_gap(new_ser[i])
                    # print(x)
            # print(self.permutations_list)
            # print(random.choice(self.permutations_list))
            new_ser = self.merge(new_ser)
            return new_ser , perm
    
    def Generate_gap(self,ser):
        
        
        (start,end) = random.choice(self.gap_lis)
        # print(start)
        # print(end)
        # print("len ",len(ser))
        new_ser = np.zeros(ser.shape)
        if end == 0:
            ser2 = ser[start:]
            new_ser[start:] = ser2
        # end = end+self.divsion_step
        else:    
            ser2 = ser[start:-end]
            new_ser[start:-end] = ser2
        # for i in range(self.segments_num):
        #     # print(ser)
        #     # print(star)
        #     # print(divsion_point)
        #     s = ser[i]
        return new_ser
        #     new_ser.append()
        
    def _generat_gap_list(self):
        
        self.gap_lis = []
        start = 0
        # self.divsion_step = x
        end = self.divsion_step
        # end = x

        for _ in range(self.divsion_step//2+1):
            for i in range(start+1,end+1):
                self.gap_lis.append((start,i))
            start +=1
            end -=1
        
        # print(self.gap_lis)
        l2 = [t[::-1] for t in self.gap_lis]
        # self.gap_lis.append(l2)
        self.gap_lis = self.gap_lis + l2 
        self.gap_lis.insert(0, (0,0))
        if self.divsion_step%2 == 0:
            self.gap_lis.append((self.divsion_step//2,self.divsion_step//2))
        random.shuffle(self.gap_lis)
        # print(self.gap_lis)
        # print(self.gap_lis[-1])
        
    def _generat_series(self,ser):
        
        new_ser = np.array([s for s in ser for i in range(self.num_of_new_copies)])
        
        return new_ser
        
    def reshape_data(self,dataset):
        
        if self.Augmented_data:
            data_permutated = np.array([row[0] for row in dataset])
            new_shape_data = (np.prod(data_permutated.shape[:-1]),data_permutated.shape[-1])
            data_permutated_new = data_permutated.reshape(new_shape_data)
            permutation_order = np.array([row[1] for row in dataset])        
            new_shape_permutation = (np.prod(permutation_order.shape[:-1]),permutation_order.shape[-1])
            permutation_order_new = permutation_order.reshape(new_shape_permutation)
        else:
            data_permutated_new = np.array([row[0] for row in dataset])
            permutation_order_new = np.array([row[1] for row in dataset])
        
        
        return data_permutated_new, permutation_order_new


        
    def generat_data(self,ser):
        new_ser = [ser[0]]
        for i in range(1,len(ser)):
            new_ser.append(random.uniform(ser[i-1], ser[i]))

            
        
        return new_ser
        
        