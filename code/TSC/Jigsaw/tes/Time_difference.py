# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 20:20:35 2023

@author: mariohabibfathi
"""



import random
import time




def _generat_permutations_list(segments_num):
    """
    This is a modified funcion from the one used in DataAug module with the intention to measures the 
    excution time for the implementation of two different algorithms and in order to show the time
    improvment between the two algorithms

    Parameters
    ----------
    segments_num (int) : The total number of segments

    Returns
    -------
    list: the list generated by the first algorithm 
    list: the list generated by the second algorithm
    """
    
    permutations_list_1 = []
    permutations_list_2 = []
    lis = list(range(0, segments_num))

    num_permutations = segments_num * 5
    i = 0
    
    star = time.time()
    while(i<num_permutations):
    

        permutation = random.sample(lis, len(lis))
        if _check_list(permutation) and permutation not in permutations_list_1 and permutation != lis:        
            permutations_list_1.append(permutation)
            i+=1
        # else:
            # print('here ',i,'perm ',permutation)
        random.shuffle(permutations_list_1)
        
        
    end = time.time()
    print(f'The total time for algorithm 1 is {end - star} seconds for segment number = {segments_num}')    
    i = 0
    star = time.time()

    while(i<num_permutations):

        permutation = _genliswithcond4(segments_num)
        if _check_list(permutation) and permutation not in permutations_list_2 and permutation != lis:        
            permutations_list_2.append(permutation)
            i+=1
        # else:
        #     print('here ',i)
        random.shuffle(permutations_list_2)
    end = time.time()
    print(f'The total time for algorithm 2 is {end - star} seconds for segment number = {segments_num}')    
    
    return permutations_list_1 , permutations_list_2






def _genliswithcond4(segments_num)    :
    """
    This is the improved function that generates the secomd algorithm which is also faster

    Parameters
    ----------
    segments_num (int) : The total number of segments


    Returns
    -------
    list: the list generated by the second algorithm

    """
    
    
    lis_genrated = [random.randint(0, 3)]
    all_elements =  list(range(0,segments_num))
    all_elements.remove(lis_genrated[0])
    k = 1 
    while (k <segments_num ):
        if k <= segments_num -2:
            candidate_num = random.randint(max(0, lis_genrated[k-1]-4), min(segments_num-1, lis_genrated[k-1]+4))
            if candidate_num not in lis_genrated :
                lis_genrated.append(candidate_num)
                if  min(all_elements) not in lis_genrated and (abs(min(all_elements)-k)>=3) :
                    lis_genrated[-1] = min(all_elements)   
                    candidate_num = min(all_elements)
                all_elements.remove(candidate_num)
                k+=1  
            check = _check_element(lis_genrated)
            if check != True:
                lis_genrated.remove(check)
                all_elements.append(check)
                k-=1
        else:
            lis_genrated.append(all_elements[0])
            k+=1            

    return lis_genrated



def _check_element(lis):  
    """
    This function checks whether the elements of the list specify the condition to conrol the randomness
    or not for second algorithm. It is a modified version of _check_list function where it returns the 
    element that broke the condition instead if False.

    Parameters
    ----------
    lis (list) : The list to be checked if it follows the condiotion or not

    Returns
    -------
    bool|int : True if it follows the condition, otherwise returns the element that breaks the condition

    """
    
    n = len(lis)
    indices = list(range(n))
    
    for i in indices:
        if abs(i - lis[i])>=4:
            return lis[i]
    return True

def _check_list(lis):  
    """
    This function checks whether the elements of the list specify the condition to conrol the randomness
    or not for first algorithm.

    Parameters
    ----------
    lis (list) : The list to be checked if it follows the condiotion or not


    Returns
    -------
    bool : True if it follows the condition, otherwise returns False.

    """
    
    
    
    
    
    
    n = len(lis)
    indices = list(range(n))
    
    for i in indices:
        if abs(i - lis[i])>=4:
            return False
    return True







list1,list2 = _generat_permutations_list(15)








