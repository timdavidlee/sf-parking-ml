from bs4 import BeautifulSoup
import numpy as np
from scrapy.selector import Selector
import pandas as pd
import json
import requests
import newspaper
from selenium.webdriver.common.keys import Keys
from selenium import webdriver
import time

# our own custom library
from common import reorder_street_block

# brings in reference dictionaries
from cleaning_data import *


def sensor_unique_street_block():
    all_ = pd.read_csv('/Users/timlee/data/sf_parking/ParkingSensorData.csv', low_memory=False)
    print(all_.shape)
    street_blocks['search'] = street_blocks['STREET_BLOCK'].map(lambda x : reorder_street_block(x))

    # output is copied and put into scraping functions
    for row in street_blocks.search.values:
        print(row)
        

def parking_process_json(path):
    with open(path,'r') as f:
        tmp = f.readlines()
    listodict = []
    for i, vv in enumerate(tmp):
        if i % 1000 == 0:
            print(i,)
        try:
            v = vv.split('|')
            lat = v[0]
            lon = v[1]
            rawjson = json.loads(v[2])
            zipc = rawjson['address']['postcode'] 
            road = rawjson['address'].get('road')
            nhood = rawjson['address'].get('neighbourhood')
            full_addr = rawjson['display_name']
            jlat = rawjson['lat']
            jlon = rawjson['lon']
            listodict.append({
                'lat': lat
                ,'lon': lon
                ,'zipcode' : zipc
                ,'road' : road
                ,'nhood' : nhood
                ,'full_addr' : full_addr
                ,'jlat':jlat
                ,'jlon':jlon
            })
        except Exception as e:
            print('error', i, e)
            break
    return(listodict)


def parking_get_json():    
    # opens all unique lat lon for parking data
    with open('./ref_data/all_lat_lon.csv', 'r') as f:
        all_lat_lon = f.read().split('\n')
        all_lat_lon = [v.split(',') for v in all_lat_lon]
        
    # for each coordinate, hit up openstreet map (with delay)
    for i, coord  in enumerate(all_lat_lon):
        lat, lon = coord
        if i % 25 == 0:
            print(i,)
        resp = requests.get('https://nominatim.openstreetmap.org/reverse?format=json&lat=%s&lon=%s&addressdetails=1' % (lat,lon))
        
        # save the response line by line
        with open ('./all_addresses2.txt', 'a') as f:
            f.write('%s|%s|%s\n' %(lat,lon,resp.content.decode('ascii', 'ignore')))

            
def parking_comb_scraped_data():
    p1 = parking_process_json('../PARK_part1.txt')
    p2 = parking_process_json('../PARK_part2.txt')
    p3 = parking_process_json('../PARK_all_addresses2.txt')
    parking_json_df = pd.DataFrame(p1 + p2 + p3)
    
    # saving 
    parking_json_df.to_csv('../ref_data/clean_parking_gps2addr.txt', sep='|', index=False)

    
def city_get_webdata():
    """
    pulls data from the website, very poorly formatted, and returns large chunks of 
    html text. there's a big blob per SF neighborhood
    """
    
    resp = requests.get('http://www.city-data.com/nbmaps/neigh-San-Francisco-California.html')
    soup = BeautifulSoup(resp.content,'html.parser')
    neighboorhoods = Selector(text=resp.content).xpath("//div[@class='neighborhood']").extract()
    text_only = []
    for neigh in neighboorhoods:
        soup = BeautifulSoup(neigh,'html.parser')
        text_only.append(soup.get_text())
    return text_only
       
    
def city_insert_newlines(text_only):
    
    metrics = ['Area','Population density','Median age', 'Housing prices', 'Population','Median household income', 'Median rent','Males:', 'Females:','Average estimated', 'Most popular']
    
    clean_text_only = []
    for tx in text_only:
        for m in metrics:
            tx = tx.replace(m,'\n-'+m)
        for k,v in replace.items():
            tx = tx.replace(k,v)
        clean_text_only.append(tx)
    return clean_text_only


def city_extract_data(clean_text_only):
    city_stats = []
    for i, section in enumerate(clean_text_only):
        output = {}
        lines = section.split('\n')
        output['neighborhood'] = lines[0].split(' ')[0]
        output['_id'] = i
        for line in lines:        
            if 'Area:' in line:
                output['area'] = line.split(':')[1].strip() 
            if 'Population:' in line:
                output['pop'] = line.split(':')[1].strip()
            if 'people per square miles' in line:
                output['ppl_density'] = line.split(':')[2].split(' ')[0]
                if output['ppl_density'] == '18,651':
                    output['ppl_density'] = line.split(':')[1].split(' ')[0]
            if ('Males:' in line) and ('years' in line):
                output['med_age'] = line.split(':')[1].strip()
            if ('Females:' in line) and ('years' in line):
                output['med_age'] = line.split(':')[1].strip()            
            if ('Males:' in line):
                output['m_pop'] = line.split(':')[1].strip()
            if 'Females:' in line:
                output['f_pop'] = line.split(':')[1].strip()
            if 'Average estimated value of detached houses in 2016' in line:
                output['house_pct'] = line.split('(')[1].split('%')[0]
                output['house_avg_value'] = line.split(':')[2].split('San')[0]
            if 'Average estimated value of townhouses' in line:
                output['twn_pct'] = line.split('(')[1].split('%')[0]
                output['twn_avg_value'] = line.split(':')[2].split('San')[0]
            if 'Average estimated value of housing units in 2-unit structures' in line:
                output['apt2_pct'] = line.split('(')[1].split('%')[0]
                output['apt2_avg_value'] = line.split(':')[2].split('San')[0]
            if "Average estimated '16 value of housing units in 3-to-4-unit structures" in line:
                output['apt4_pct'] = line.split('(')[1].split('%')[0]
                output['apt4_avg_value'] = line.split(':')[2].split('San')[0]

        city_stats.append(output)
    return city_stats


def clean_dollars(x):
    """
    Used to clean up dollar fields of $ and commas 
    """
    if type(x) != float:
        return x.replace('$','').replace(',','').replace('city','')

def clean_age(x):
    """
    Used to clean up age fields such as '35 years'
    """
    if type(x) != float:
        return x.replace('years','')
    

def city_stats_to_df(city_stats):
    """
    takes in list of dictionaries (per row
    converts to dataframe, cleans the columns
    and converts types
    """
    cs = pd.DataFrame(city_stats)
    
    # cleaning money
    cs['twn_avg_value'] = cs['twn_avg_value'].map(clean_dollars)
    cs['apt2_avg_value'] = cs['apt2_avg_value'].map(clean_dollars)
    cs['apt4_avg_value'] = cs['apt4_avg_value'].map(clean_dollars)
    
    # cleaning area
    cs['area'] = cs['area'].str.replace('square miles','')
    cs['area'] = cs['area'].map(lambda x : x.split(' ')[0])
    cs['house_avg_value'] = cs['house_avg_value'].map(clean_dollars)
    
    # cleaning age
    cs['f_pop'] = cs['f_pop'].map(clean_age)
    cs['m_pop'] = cs['m_pop'].map(clean_age)
    cs['med_age'] = cs['med_age'].map(clean_age)
    cs['pop'] = cs['pop'].map(clean_dollars)
    
    # converting types + saving
    num_cols = [col for col in cs.columns if col != 'neighborhood'] 
    for col in num_cols:
        print(col)
        cs[col] = cs[col].astype(float)
    cs.to_csv('../ref_data/nh_city_stats.txt', sep='|')
    
    
    
def addr2gps_intersection_search():
    """
    Works off the unique sensor data locations
    submits addresses
    pulls gps coord
    then repulls address to get zip code
    """
    
    offset = 0
    for i, word_addr  in enumerate(tt_word_addr[offset:]):
        
        #submit address
        address_box = driver.find_element_by_xpath("//input[@id='address']")
        address_box.clear()
        address_box.send_keys(word_addr + ', San Francisco')
        address_box.send_keys(Keys.ENTER)
        submit = driver.find_element_by_xpath("//div[@class='form-group'][2]/div/button")
        submit.click()
        time.sleep(1)
        
        # get latlong
        address_field = driver.find_element_by_id('address')
        latbox = driver.find_element_by_xpath("//input[@id='latitude']")
        lat_val = latbox.get_attribute('value')
        lonbox = driver.find_element_by_xpath("//input[@id='longitude']")
        lon_val = lonbox.get_attribute('value')
        
        # resubmit to get Zip code
        submit = driver.find_element_by_xpath("//div[@class='form-group'][3]/div/button")
        submit.click()
        time.sleep(1)
        addr_val = address_field.get_attribute('value')
        
        #store + print values
        ad1 = word_addr.split(' and ')[0]
        ad2 = word_addr.split(' and ')[1]
        zipc = addr_val[-10:-5]
        output = [str(i+offset), ad1, ad2, lat_val, lon_val, zipc]
        print('|'.join(output))
        

def addr2gps_single_search():
    offset = 0
    for i, word_addr  in enumerate(snsr_loc[offset:]):
        address_box = driver.find_element_by_xpath("//input[@id='address']")
        address_box.clear()
        address_box.send_keys(word_addr + ', San Francisco')
        address_box.send_keys(Keys.ENTER)
        submit = driver.find_element_by_xpath("//div[@class='form-group'][2]/div/button")
        submit.click()
        time.sleep(1)
        address_field = driver.find_element_by_id('address')
        latbox = driver.find_element_by_xpath("//input[@id='latitude']")
        lat_val = latbox.get_attribute('value')
        lonbox = driver.find_element_by_xpath("//input[@id='longitude']")
        lon_val = lonbox.get_attribute('value')
        submit = driver.find_element_by_xpath("//div[@class='form-group'][3]/div/button")
        submit.click()
        time.sleep(1)
        addr_val = address_field.get_attribute('value')
        ad1 = word_addr
        zipc = addr_val[-10:-5]
        output = [str(i+offset), ad1, lat_val, lon_val, zipc]
        print('|'.join(output))
        
        
# bad zip code handling

def resubmit_missing_zips(follow_up):
    missingzips = []
    for id_, lat, lon in follow_up:
        latbox = driver.find_element_by_xpath("//input[@id='latitude']")
        latbox.clear()
        latbox.send_keys(str(lat))
        lonbox = driver.find_element_by_xpath("//input[@id='longitude']")
        lonbox.clear()
        lonbox.send_keys(str(lon))
        submit = driver.find_element_by_xpath("//div[@class='form-group'][3]/div/button")
        submit.click()
        time.sleep(1)
        address_field = driver.find_element_by_id('address')
        addr_val = address_field.get_attribute('value')

        print(id_, lat, lon ,addr_val)
        missingzips.append([id_, lat, lon ,addr_val])
    
    return missingzips

def resubmit_for_badzip()
    df = pd.read_csv('../ref_data/sensor_2_gps.txt', delimiter = '|', header=None)
    df.columns = ['id', 'street', 'lat','lon','zip']
    follow_up = df[df['zip']== 'ed ad'][['id','lat','lon']].values

    missingzips = resubmit_missing_zips(follow_up)
        
    miss_df = pd.DataFrame(missingzips)
    miss_df = miss_df[miss_df.iloc[:,3] != 'No resolved address'].copy()
    miss_df['id'] = miss_df.iloc[:,0].astype(int)
    miss_df['zip'] = miss_df.iloc[:,3].str[-10:-5]
    miss_df.index = miss_df['id']
    miss_df = miss_df['zip'].copy()
    df.loc[df['zip']== 'ed ad','zip'] = df.loc[df['zip']== 'ed ad','id'].map(miss_df)
    follow_up = df[df['zip'].isna()][['id','lat','lon']].values
   
    while len(follow_up) >0:
        
        rmissingzips = resubmit_missing_zips(follow_up)
        
        miss_df = pd.DataFrame(missingzips)
        miss_df = miss_df[miss_df.iloc[:,3] != 'No resolved address'].copy()
        miss_df['id'] = miss_df.iloc[:,0].astype(int)
        miss_df['zip'] = miss_df.iloc[:,3].str[-10:-5]
        miss_df.index = miss_df['id']
        miss_df = miss_df['zip'].copy()
        df.loc[df['zip'].isna(),'zip'] = df.loc[df['zip'].isna(),'id'].map(miss_df)
        follow_up = df[df['zip'].isna()][['id','lat','lon']].values
