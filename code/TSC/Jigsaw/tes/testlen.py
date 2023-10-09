#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 16:51:24 2023

@author: mariohabibfathi
"""

import itertools
import random
import time

# def generat_permutations_list(segments_num):
    
#     permutations_list = []
#     if segments_num < 2:
#         segments_num = 2
#     lis = list(range(0, segments_num))
    
#     if segments_num < 4:
#         permutations_list= list(itertools.permutations(lis))
#         permutations_list.pop(0)
#         random.shuffle(permutations_list)        
    
    
#     else:
#         num_permutations = segments_num * 5
#         i = 0
#         while(i<num_permutations):
#             if segments_num < 15:
#                 permutation = random.sample(lis, len(lis))
#                 if (check_list(permutation)) and (permutation not in permutations_list or permutation not in lis):        
#                     permutations_list.append(permutation)
#                     i+=1
#                 else:
#                     print('here ',i)
#                 random.shuffle(permutations_list)
#             else:
#                 random_list = []
#                 max_distance = 4 # define the maximum distance allowed between a number and its initial position
#                 # Generate the first number
#                 first_num = random.randint(0, max_distance-1)
#                 random_list.append(first_num)
#                 n = segments_num
#                 for k in range(1, n):
#                     while True:
#                         candidate_num = random.randint(max(0, random_list[k-1]-max_distance), min(n-1, random_list[k-1]+max_distance))
#                         if candidate_num not in random_list :
#                             random_list.append(candidate_num)
#                             break
#                         else:
#                             print('here ',i,' per lis ',random_list)
#                 permutations_list.append(random_list)
#                 i +=1
                
                
                
#     return permutations_list


def generat_permutations_list(segments_num):
    
    permutations_list = []
    if segments_num < 2:
        segments_num = 2
    lis = list(range(0, segments_num))
    
    if segments_num < 4:
        permutations_list= list(itertools.permutations(lis))
        permutations_list.pop(0)
        random.shuffle(permutations_list)        
    
    
    else:
        num_permutations = segments_num * 5
        i = 0
        while(i<num_permutations):
            if segments_num < 15:
                permutation = random.sample(lis, len(lis))
                if (check_list(permutation)) and (permutation not in permutations_list or permutation not in lis):        
                    permutations_list.append(permutation)
                    i+=1
                else:
                    print('here ',i)
                random.shuffle(permutations_list)
            else:
                random_list = []
                max_distance = 4
                first_num = random.randint(0, max_distance-1)
                random_list.append(first_num)
                n = segments_num
                first_elem = 0
                last_elem = max_distance-1
                check_lis = range(first_elem,last_elem)
                k = 1 
                while (k <n ):
                    print('k',k)
                    # print(type(random_list[k-1]))
                    print(random_list)
                    # print(min(n-1, random_list[k-1]+max_distance+2))
                    candidate_num = random.randint(max(0, random_list[k-1]-max_distance), min(n-1, random_list[k-1]+max_distance+2))
                    if candidate_num not in random_list :
                        random_list.append(candidate_num)
                        k+=1
                        if len(random_list)>3 and min(check_lis) not in random_list :
                            random_list[-1] = min(check_lis)                       

                 
                
                
                    # while True:
                    #         break
                    # else:
                    #     print('here ',i,' per lis ',random_list)
                    first_elem += 1
                    last_elem += 1
                    check_lis = range(first_elem,last_elem)
                permutations_list.append(random_list)
                i +=1
                
                
                
    return permutations_list
def check_list(lis):  
    n = len(lis)
    indices = list(range(n))
    
    for i in indices:
        if abs(i - lis[i])>=4:
            return False
    return True


def gelis(segments_num):
    random_list = []
    max_distance = 4
    first_num = random.randint(0, max_distance-1)
    random_list.append(first_num)
    k = 1 
    while (k <segments_num ):
        # print('k',k)
        # print(type(random_list[k-1]))
        # print(random_list)
        # print(min(n-1, random_list[k-1]+max_distance+2))
        # candidate_num = random.randint(0, segments_num-1)
        first_elem = 0
        last_elem = max_distance-1
        check_lis = list(range(first_elem,last_elem))
        candidate_num = random.randint(max(0, random_list[k-1]-max_distance), min(segments_num-1, random_list[k-1]+max_distance+2))
        if candidate_num not in random_list :
            random_list.append(candidate_num)
            k+=1
            if len(random_list)>=3 and min(check_lis) not in random_list :
                random_list[-1] = min(check_lis)                  
    return random_list
    
    
    
    
    
def genlisnocond(segments_num)    :
    lis_genrated = [random.randint(0, 3)]
    all_elements =  list(range(0,segments_num))
    first_elem = 0
    last_elem = 3
    check_lis = list(range(first_elem,last_elem))
    k = 1 
    while (k <segments_num ):
        if len(all_elements) != 1:
            
            # print('k',k)
            # print(type(random_list[k-1]))
            # print(random_list)
            # print(min(n-1, random_list[k-1]+max_distance+2))
            # candidate_num = random.randint(0, segments_num-1)
            candidate_num = random.randint(0, segments_num-1)
            print(all_elements)
            print(candidate_num)
            if candidate_num not in lis_genrated :
                lis_genrated.append(candidate_num)
                all_elements.remove(candidate_num)
                k+=1            
        else:
            print(k)
            lis_genrated.append(all_elements)
    return lis_genrated
  


def genliswithcond(segments_num)    :
    lis_genrated = [random.randint(0, 3)]
    all_elements =  list(range(0,segments_num))
    first_elem = 0
    last_elem = 3
    check_lis = list(range(first_elem,last_elem))
    k = 1 
    while (k <segments_num ):
        if len(all_elements) != 1:
            print()
            # print('k',k)
            # print(type(random_list[k-1]))
            # print(random_list)
            # print(min(n-1, random_list[k-1]+max_distance+2))
            # candidate_num = random.randint(0, segments_num-1)
            candidate_num = random.randint(max(0, lis_genrated[k-1]-4), min(segments_num-1, lis_genrated[k-1]+4))
            # print('all ',all_elements)
            # print('Initial number ',candidate_num)
            # print('Generated list ',lis_genrated)
            # print('max ',max(first_elem, lis_genrated[k-1]-4))
            # print('min ',min(last_elem, lis_genrated[k-1]+4))
            if candidate_num not in lis_genrated :
                lis_genrated.append(candidate_num)
                if len(lis_genrated)>=3 and min(check_lis) not in lis_genrated :
                    lis_genrated[-1] = min(check_lis)   
                    first_elem +=1
                    last_elem +=1
                    candidate_num = min(check_lis)
                    check_lis = list(range(first_elem,last_elem))
                print('Before remove ',candidate_num)
                all_elements.remove(candidate_num)
                k+=1            
        else:
            lis_genrated.append(all_elements)
    return lis_genrated
    
def genliswithcond2(segments_num)    :
    lis_genrated = [random.randint(0, 3)]
    all_elements =  list(range(0,segments_num))
    all_elements.remove(lis_genrated[0])
    first_elem = 0
    last_elem = 3
    check_lis = list(range(first_elem,last_elem))
    k = 1 
    while (k <segments_num ):
        if len(all_elements) != 1:
            # print()
            # print('k',k)
            # print(type(random_list[k-1]))
            # print(random_list)
            # print(min(n-1, random_list[k-1]+max_distance+2))
            # candidate_num = random.randint(0, segments_num-1)
            candidate_num = random.randint(max(0, lis_genrated[k-1]-4), min(segments_num-1, lis_genrated[k-1]+4))
            print('all remaning ',all_elements)
            print('Initial number ',candidate_num)
            print('Generated list ',lis_genrated)
            print('In check now  ',check_lis)
            print('max ',max(first_elem, lis_genrated[k-1]-4))
            print('min ',min(last_elem, lis_genrated[k-1]+4))
            if candidate_num not in lis_genrated :
                lis_genrated.append(candidate_num)
                if len(lis_genrated)%3>1 and min(check_lis) not in lis_genrated :
                    lis_genrated[-1] = min(check_lis)   
                    candidate_num = min(check_lis)
                
                for item in lis_genrated:
                    while item in check_lis:
                        check_lis.remove(item)
                # check_lis = [item for item in list1 if item not in list2]
                # if candidate_num in check_lis:
                #     check_lis.remove(candidate_num)
                if last_elem <segments_num:                
                    check_lis.append(last_elem)
                    last_elem+=1
                # print('Before remove ',candidate_num)
                all_elements.remove(candidate_num)
                k+=1            
        else:
            # print('here')
            # print('all remaning ',all_elements)
            # print('all remaning ',all_elements[0])
            print('Generated list ',lis_genrated)
            print(k)
            lis_genrated.append(all_elements[0])
            return lis_genrated
    

# def generate_permuted_list(n):
#     lst = list(range(n))
#     positions = list(range(n))

#     # Generate a random permutation of the list by swapping pairs of elements
#     for i in range(n):
#         j = random.randint(max(0, i-4), min(n-1, i+4))
#         while j == i or abs(positions[j] - positions[i]) > 4:
#             j = random.randint(max(0, i-4), min(n-1, i+4))
#         lst[i], lst[j] = lst[j], lst[i]
#         positions[i], positions[j] = positions[j], positions[i]

#     # Ensure that no element is in the same position as its initial position
#     for i in range(n):
#         if lst[i] == i:
#             # Find a random index that is different from the element's initial position
#             j = random.randint(0, n-1)
#             while j == i:
#                 j = random.randint(0, n-1)
#             # Swap the element with the randomly chosen index
#             lst[i], lst[j] = lst[j], lst[i]

#     return lst
# import random


def generate_permuted_list(n):
    lst = list(range(n))
    random.shuffle(lst)
    for i in range(n):
        valid_positions = [j for j in range(max(0, i-3), min(n, i+4)) if j != i and abs(j - lst.index(lst[i])) >= 1 and abs(j - lst.index(lst[i])) <= 4]
        if not valid_positions:
            return generate_permuted_list(n)
        target_position = random.choice(valid_positions)
        lst[i], lst[target_position] = lst[target_position], lst[i]
    return lst


    
def genliswithcond3(segments_num)    :
    lis_genrated = [random.randint(0, 3)]
    all_elements =  list(range(0,segments_num))
    all_elements.remove(lis_genrated[0])
    # first_elem = 0
    # last_elem = 3
    # check_lis = list(range(first_elem,last_elem))
    k = 1 
    while (k <segments_num ):
        if k <= segments_num -2:
            # print()
            # print('k',k)
            # print(type(random_list[k-1]))
            # print(random_list)
            # print(min(n-1, random_list[k-1]+max_distance+2))
            # candidate_num = random.randint(0, segments_num-1)
            candidate_num = random.randint(max(0, lis_genrated[k-1]-4), min(segments_num-1, lis_genrated[k-1]+4))
            # print('all remaning ',all_elements)
            # print('Initial number ',candidate_num)
            # print('Generated list ',lis_genrated)
            # print('In check now  ',check_lis)
            # print('max ',max(first_elem, lis_genrated[k-1]-4))
            # print('min ',min(last_elem, lis_genrated[k-1]+4))
            if candidate_num not in lis_genrated :
                lis_genrated.append(candidate_num)
                if  min(all_elements) not in lis_genrated and (abs(min(all_elements)-k)>=3) :
                    lis_genrated[-1] = min(all_elements)   
                    candidate_num = min(all_elements)
                
                # for item in lis_genrated:
                #     while item in check_lis:
                #         check_lis.remove(item)
                # check_lis = [item for item in list1 if item not in list2]
                # if candidate_num in check_lis:
                #     check_lis.remove(candidate_num)
                # if last_elem <segments_num:                
                #     check_lis.append(last_elem)
                #     last_elem+=1
                # print('Before remove ',candidate_num)
                all_elements.remove(candidate_num)
                k+=1            
        else:
            print('here')
            print('all remaning ',all_elements)
            # print('all remaning ',all_elements[0])
            print('Generated list ',lis_genrated)
            print(k)
            lis_genrated.append(all_elements[0])
            k+=1            

    return lis_genrated



def genliswithcond4(segments_num)    :
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
            check = check_element(lis_genrated)
            if check != True:
                lis_genrated.remove(check)
                all_elements.append(check)
                k-=1
        else:
            lis_genrated.append(all_elements[0])
            k+=1            

    return lis_genrated


def check_element(lis):  
    n = len(lis)
    indices = list(range(n))
    
    for i in indices:
        if abs(i - lis[i])>=4:
            return lis[i]
    return True


# start = time.time()
# lis = genlisnocond(15)

# print('total time: ', time.time() - start)
# for i in range(100):
    
    
count = 0 
total_time = time.time()
for i in range(1000):
    
    print()
    start = time.time()
    lis2 = genliswithcond4(15)
    
    print('total time: ', time.time() - start)
    print(check_list(lis2))
    if check_list(lis2) != True:
        count+=1
print('count ',count)
print('total time: ', time.time() - total_time)



# start = time.time()
# lis2 = genliswithcond4(15)

# print('total time: ', time.time() - start)
# print(check_list(lis2))


















































# import random

# def genliswithcond3(segments_num):
#     lis_generated = [random.randint(0, 3)]
#     all_elements = list(range(segments_num))
#     all_elements.remove(lis_generated[0])
#     first_elem = 0
#     last_elem = 3
#     check_lis = list(range(first_elem, last_elem))
#     k = 1 
    
#     while k < segments_num:
#         if len(all_elements) != 1:
#             candidate_num = random.randint(max(0, lis_generated[k-1]-4), min(segments_num-1, lis_generated[k-1]+4))
#             if candidate_num not in lis_generated:
#                 if len(lis_generated) % 3 > 1 and min(check_lis) not in lis_generated:
#                     lis_generated[-1] = min(check_lis)
#                     candidate_num = min(check_lis)
#                 lis_generated.append(candidate_num)
#                 for item in lis_generated:
#                     while item in check_lis:
#                         check_lis.remove(item)
#                 if last_elem < segments_num:                
#                     check_lis.append(last_elem)
#                     last_elem += 1
#                 if candidate_num in all_elements:
#                     all_elements.remove(candidate_num)
#                 k += 1
#             else:
#                 continue
#         else:
#             remaining_num = all_elements[0]
#             if abs(remaining_num - lis_generated[-1]) > 4:
#                 lis_generated.append(lis_generated[-1] + 1)
#             else:
#                 lis_generated.append(remaining_num)
#             k += 1
#     return lis_generated
    

# def genliswithcond4(segments_num):
#     lis_generated = [random.randint(0, 3)]
#     all_elements = list(range(segments_num))
#     all_elements.remove(lis_generated[0])
#     k = 1
    
#     while k < segments_num:
#         if len(all_elements) != 1:
#             # generate candidate numbers based on previously generated numbers
#             candidates = []
#             for i in range(-4, segments_num+4):
#                 candidate_num = lis_generated[k-1] + i
#                 if candidate_num in all_elements:
#                     candidates.append(candidate_num)
#             # choose candidate with minimum distance to previously generated number
#             candidate_num = min(candidates, key=lambda x: abs(x - lis_generated[k-1]))
#             lis_generated.append(candidate_num)
#             all_elements.remove(candidate_num)
#             k += 1
#         else:
#             lis_generated.append(all_elements[0])
#             k += 1
#     return lis_generated





# def genliswithcond5(segments_num):
#     lis_genrated = [random.randint(0, segments_num-1)]
#     all_elements = list(range(segments_num))
#     all_elements.remove(lis_genrated[0])
    
#     for i in range(1, segments_num):
#         prev_value = lis_genrated[-1]
#         candidate_num = random.randint(max(0, prev_value-4), min(segments_num-1, prev_value+4))
#         while candidate_num in lis_genrated or abs(candidate_num - prev_value) > 4:
#             candidate_num = random.randint(max(0, prev_value-4), min(segments_num-1, prev_value+4))
#         lis_genrated.append(candidate_num)
#         all_elements.remove(candidate_num)
        
#     return lis_genrated


# import random

# def genliswithcond6(segments_num):
#     lis_generated = [random.randint(0, 3)]
#     all_elements = list(range(0, segments_num))
#     all_elements.remove(lis_generated[0])
#     last_elem = 3
#     check_lis = list(range(last_elem - 2, last_elem + 1))
#     k = 1
    
#     while k < segments_num:
#         if len(all_elements) > 1:
#             candidate_num = random.randint(max(0, lis_generated[k-1]-4), min(segments_num-1, lis_generated[k-1]+4))
#             if candidate_num not in lis_generated:
#                 lis_generated.append(candidate_num)
#                 if len(lis_generated) % 3 > 1 and min(check_lis) not in lis_generated:
#                     lis_generated[-1] = min(check_lis)
#                     candidate_num = min(check_lis)

#                 for item in lis_generated:
#                     while item in check_lis:
#                         check_lis.remove(item)

#                 all_elements.remove(candidate_num)
#                 k += 1

#                 # check if remaining segments can be added without violating max distance constraint
#                 remaining_segments = segments_num - k
#                 if remaining_segments <= 4:
#                     for i in range(remaining_segments):
#                         candidate_num = random.choice(all_elements)
#                         lis_generated.append(candidate_num)
#                         all_elements.remove(candidate_num)
#                         k += 1
#                     break
#             else:
#                 continue
#         else:
#             lis_generated.append(all_elements[0])
#             break
    
#     return lis_generated



# 
















# print(random.randint(0,3))

# import random

# n = 10 # define the length of the list

# # Generate the rest of the numbers

# print(random_list) # print the list of unique random numbers with a maximum distance of 4 from their initial positions
















