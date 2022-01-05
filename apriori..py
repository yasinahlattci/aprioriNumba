import numpy as np
from numba import jit, njit
from copy import deepcopy
import sqlite3
import pandas as pd
#from timeit import timeit
from time import time


def ap(database, min_sup):

    def transform_elements(elements):
        transform_dict = dict()
        transfer_number = 1
        for i in range(len(elements)):
            for j in range(len(elements[i])):
                if elements[i][j] in transform_dict.keys():
                    elements[i][j] = transform_dict[elements[i][j]]
                else:
                    transform_dict[elements[i][j]] = transfer_number
                    transfer_number += 1
                    elements[i][j] = transform_dict[elements[i][j]]
        len_max = max(len(x) for x in elements)
        for i in range(len(elements)):
            dif = len_max - len(elements[i])
            if len(elements[i]) < len_max:
                elements[i].extend([-1] * dif)
        inv_dict = {v: k for k, v in transform_dict.items()}
        return inv_dict, np.array(elements, np.int64)
    ##########################################################################
    def back_transform(inverse_dict, output):
        for i in range(len(output)):
            for j in range(len(output[i][0])):
                output[i][0][j] = inverse_dict[output[i][0][j]]
        return output
    ##########################################################################
    @njit()
    def minsup_control(element_sup, min_sup):
        if element_sup >= min_sup:
            return True
        else:
            return False
    ##########################################################################
    @njit()
    def birlestir(el1, el2):
        return np.append(el1, el2)

    ##########################################################################
    @njit()
    def stack(element, item_list):
        index_element = element[-1]
        stack_element = np.zeros((item_list.shape[0] - index_element), np.int32)
        ct = 0
        for i in range(index_element, item_list.shape[0]):

            stack_element[ct] = item_list[i]
            ct += 1
        return stack_element
    ##########################################################################
    @njit()
    def frequent_finder(element, list1, len_main):
        ct = np.zeros(list1.shape[0], np.int64)
        ct_num = np.zeros((list1.shape[0], element.shape[0]), np.int64)
        for i in range(list1.shape[0]):
            for m in range(len(element)):
                for j in range(list1.shape[1]):
                    ct_num[i, m] = [1 if list1[i, j] == element[m] else 0][0]
                    if ct_num[i, m] == 1:
                        break

            if np.all(ct_num[i]):
                ct[i] = 1
        # return ct
        return np.sum(ct) / len_main
    ##########################################################################
    condition = True
    transfer_db = deepcopy(database)
    inverse_dict, transformed_list = transform_elements(transfer_db)
    itemlist = np.array(list(inverse_dict.keys()), np.int64)
    len_main = transformed_list.shape[0]
    sonuc_listesi = []
    for i in range(itemlist.shape[0]):
        m = frequent_finder(np.array([itemlist[i]], np.int64), transformed_list, len_main)
        if minsup_control(m, min_sup) == True:
            sonuc_listesi.append([[itemlist[i]], m])
    current_list = deepcopy(sonuc_listesi)
    while condition == True:
        new_list = list()
        for i, j in current_list:
            t = stack(np.array(i, np.int64), itemlist)
            for m in t:
                new_element = birlestir(np.array([i], np.int64), np.array([m], np.int64))
                sup = frequent_finder(np.array(new_element, np.int64), transformed_list, len_main)
                if minsup_control(sup, min_sup) == True:
                    sonuc_listesi.append([list(new_element), sup])
                    new_list.append([list(new_element), sup])
        del current_list
        current_list = deepcopy(new_list)
        del new_list
        if len(current_list) <= 1:
            condition = False
    return sonuc_listesi

def import_db(baglanti_adi):
    con = sqlite3.connect(baglanti_adi)
    ## İLK TABLE I ALIYORUZ
    cursor = con.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    tables = tables[0][0]
    ## İLK TABLE I DATABASE E ATIYORUZ
    database = pd.read_sql_query(f"SELECT * from {tables}", con)
    ##SADECE DATANIN KAYITLI OLDUĞU SÜTUNU ALIYORUZ. NUMARALANDIRMA SÜTUNU ZATEN PANDAS TAN GELECEK.
    sütun = len(database.columns)
    if sütun > 1:
        for i in range(sütun - 1):
            database = database.drop(database.columns[0], axis=1)
    database = database.values.tolist()
    for num, element in enumerate(database):
        database[num] = element[0].split(',')
    return database

adres = r"C:\Users\ysnah\Desktop\TEZ\TezV1.1\databases\food1000x.db"
database = import_db(adres)
cur_time = time()
sonuclar = ap(database, 0.1)
print("Süre:",time()-cur_time)