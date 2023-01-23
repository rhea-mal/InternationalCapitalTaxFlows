import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from copy import deepcopy
import random
from statistics import median,mean
import csv
from natsort import natsorted
import pickle
import os
pd.options.mode.chained_assignment = None  # default='warn'
import openpyxl
import math


def get_currencies(df):
    new_df = df
    for j in range(len(df)):
        rate = str(df['Rate'][j])
        first = 0
        last = 0
        for i in range(len(rate)):
            if rate[i] == "(":
                first = i + 1
            if rate[i:i+3] == 'to ':
                last = i-1
        new_rate = rate[first:last].lower().strip().replace(".", "")
        new_df['Currency'][j] = new_rate
    return new_df

def error_check(df):
    df['City'] = 0
    df['TotalName'] = 0
    df['Errors'] = ""
    banks = []
    total = []
    new_df = df
    for bank in df['BankName']:
        if ((type(bank) == str) or (not math.isnan(bank))):
            banks.append(bank)

    for i in range(len(df)):
        city = str(df['LocationTBC'][i]).split(',')
        if len(city) >= 2:
            df['City'][i] = city[-2] + ',' + city[-1]
        else:
            if ((type(df['BankName'][i]) == str) or (not math.isnan(df['BankName'][i]))):
                new_df['City'][i] = df['LocationTBC'][i]
                new_df['Errors'][i] = 'Incomplete Address'
        new_df['TotalName'][i] = str(df['BankName'][i]) + ' , ' + str(df['City'][i])    
                
    for bank in df['TotalName']:
        if ((type(bank) == str) or (not math.isnan(bank))):
            total.append(bank)
    return new_df

def getTotal(df):
    total = []
    for bank in df['TotalName']:
        if ((type(bank) == str) or (not math.isnan(bank))):
            total.append(bank)
    return total

## clean dataframe, remove empty rows 
def clean_sheet(df):
    df_new = pd.DataFrame(columns = df.columns)
    for j in range(len(df)):
        ## Gets rid of empty rows ??
        if (type(df['Currency'][j]) == str):
            new_row = df.iloc[j]
            df_new.loc[len(df_new)] = new_row
            # remove commas
            if (df['Item'][j] != None and "," in str(df['Item'][j])):
                df_new['Item'][len(df_new) - 1] = df['Item'][j].replace(',', '').strip()
    return df_new

## ASSEMBLES BANK DICT
def assemble_bank_dict(df, total):
    dict = {}
    prev = 0
    for i in range(1, len(total)):
        if total[i]:
            first_row = prev
            if i == len(df)-1:
                last_row = len(df['BankName']) - 1
            else:
                last_row = i - 1
                prev = i
        
        dict[total[i]] = [first_row, last_row]
    return dict

def new_check(df, dict, total):
    new_df = df;
    for bank in dict:
        for i in range(dict[bank][0], dict[bank][1]):
            if (df['Currency'][i] != df['Currency'][i+1]):
                new_df['Errors'][i] += ' Currency Error'
            if df['Year'][i] == 'missing':
                new_df['Errors'][i] += 'Missing Year Error'
            else:
                for i in range(dict[bank][0], dict[bank][1]):
                    if (abs(int(df['Year'][i]) - int(df['Year'][i+1])) > 1):
                        new_df['Errors'][i] += 'Year Value Error'
                    if df['Item'][i] == 'Total' and df['Item Type'][i] == 'L':
                        total_L = df['Value'][i]
                    if df['Item'][i] == 'Total' and df['Item Type'][i] == 'A':
                        total_A = df['Value'][i]
                        if total_L != total_A:
                            new_df['Errors'][i] += 'Total Value Error'
            if df['Country'][i] == 'NaN':
                new_df['Errors'][i] += 'No Country Error'
    return new_df

def currency_check(df, cur):
    df_new = df
    df_new['CurrencyCheck'] = 'missing'
    for i in range(len(df)):
        elem = df['Currency'][i].lower().replace(".", "")
        if df['Currency'][i].lower() in cur['Currency']:
            df_new['CurrencyCheck'][i] = df['Currency'][i]
    return df_new

def country_check(df):
    df_new = df
    for i in range(len(df)):
        if (type(df['Rate'][i]) == str):
            end = df['Rate'][i].find('(')
            val = df['Rate'][i][:end].replace(".", "").strip()
            df_new['Rate'][i] = val
    return df_new


def match_ISO(df, ISO):
    df_new = df
    ISO['lower_cur'] = ISO['Currency'].str.lower()
    df_new['ISO_code'] = ''
    df_new['1959 (avg)'] = ''
    df_new['1958 (avg)'] = ''
    df_new['1957 (avg)'] = ''
    for i in range(len(df)):
        country = df['Country'][i]
        if (type(country) == str):
            idx = tuple(ISO.index[ISO['Country'] == country])
            if len(idx):
                idx = idx[0]
                df_new['ISO_code'][i] = ISO['Code'][idx]
                df_new['1959 (avg)'][i] = ISO['1959 (avg)'][idx]
                df_new['1958 (avg)'][i] = ISO['1958 (avg)'][idx]
                df_new['1957 (avg)'][i] = ISO['1957 (avg)'][idx]
        else:
            cur_val = df['Currency'][i].lower()
            if (type(cur_val) == str):
                idx = tuple(ISO.index[ISO['Currency'] == cur_val])
                if len(idx):
                    idx = idx[0]
                    df_new['ISO_code'][i] = ISO['Code'][idx]
                    df_new['1959 (avg)'][i] = ISO['1959 (avg)'][idx]
                    df_new['1958 (avg)'][i] = ISO['1958 (avg)'][idx]
                    df_new['1957 (avg)'][i] = ISO['1957 (avg)'][idx]
    return df_new

def match_macro(df, macro):
    df_new = df
    df_new['Macro_xr'] = ''
    for i in range(len(df)):
        country = df['Country'][i]
        year = df['Year'][i]
        if (type(country) == str and year != 'missing'):
            idx = (macro[(macro['country']  == country) & (macro['year'] == int(year))].index.tolist())
            if idx:
                df_new['Macro_xr'][i] = float(macro['xrusd'][idx])
    return df_new