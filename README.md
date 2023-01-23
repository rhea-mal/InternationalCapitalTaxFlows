# International Capital Tax Flows
Work with Dr. Antonio Coppola &amp; Dr. Chenzi Xu w. Stanford GSB/SIEPR

#  Balance Sheet & Exchange Rate Clean

This repository contains the code to clean digitized balance sheets and combine tables with their corresponding exchange rates in USD and GBP. This notebook outlines functions to clean tables from common typos from digitization. 

## Inputs

The code requires 5 input filepaths:
EXCEL_PATH = file path to the excel form of balance sheet to be processed
RATES_PATH = file path to excel of exchange rates (from back of the book) with  rates over 3 year interval. Must be from currency into pound sterling.
ISO_PATH = filepath to excel file of currencies and countries & their corresponding ISO codes. 
MACRO_PATH = the filepath to the macrohistory.net database sheet to cross reference exchange rates
OUTPUT_PATH = where to store the final cleaned and combined balance sheet
Note: Still missing USD to GBP data set list


## Workflows

### Beta Deployment

The cleaning pipeline first takes in the appropriate sheets and cleans the balance sheet code. Digitized balance sheets and exchange rate tables from the almanac must be available as excel sheets. The ISO sheet was generated and manually adjusted in a separate code from the digitized exhchange rate data. Note that not all rates are from X currency to GBP, instead some may be reported inverted. These cases may be overlooked by the code.

All sheets are combined by country, currency, or ISO code. Appropriate exchange rate columns are appended to the end of the excel file and output in OUTPUT_PATH.

### Contributors

This project is part of the Banking Almanac project of Three Centuries of International Capital Flows with Dr. Antonio Coppolla & Dr. Chenzi Xu (https://github.com/AntonioCoppola/coppola-xu). 
