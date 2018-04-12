import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import holidays
import os
import _pickle as pickle
from sklearn.model_selection import train_test_split
from multiprocessing import cpu_count, Pool
from my_decorators import *

US_HOLIDAYS = holidays.UnitedStates()
DATA_PATH = '/Users/timlee/data/sf_parking/'
REPO_PATH = '/Users/timlee/myrepos/final-project-chengcheng-tim/'
REF_PATH = '/Users/timlee/myrepos/final-project-chengcheng-tim/ref_data/'


def load_clean_parking():
    return pd.read_feather(REF_PATH + 'matched_parking_w_loc_id.feather')


def load_clean_train():
    """
    loads training data with clean street names
    and midpoint gps
    """
    return pd.read_feather(REF_PATH + 'df_train_w_clean_str_gmaps_addr.feather')

def load_midpoint_id():
    with open(REF_PATH + 'mid_gps_id_4_yelp.pkl', 'rb') as f:
        mid2gps = pickle.load(f)
    return mid2gps

def load_clean_test():
    """
    loads training data with clean street names
    and midpoint gps
    """
    return pd.read_feather(REF_PATH + 'df_test_w_clean_str_gmaps_addr.feather')


@timethis
def get_train():
    """
    Loads training data
    """
    df_train = pd.read_csv(DATA_PATH + 'train-parking.csv', parse_dates=['Date'])
    return df_train


@timethis
def get_test():
    """
    Loads test data
    """
    df_test = pd.read_csv(DATA_PATH +'test-no-label-parking.csv', parse_dates=['Date'])
    return df_test


@timethis
def get_sensor():
    """
    loads sensor data
    """
    sn_df = pd.read_feather(DATA_PATH+'P_sensor.feather')
    return sn_df


def plot_pred_coord(y_true, lat_true, lon_true, y_pred):
    """
    Should have ANY_SPOT, LAT, LON in both datasets
    makes a 2d plot to show what is right and wrong
    """
    plot_mask1 = (y_hat == y_vl)
    plot_mask2 = (y_hat != y_vl)
    plt.figure(figsize=(11,6))
    plt.scatter(lat_true[plot_mask1], lon_true[plot_mask1], c = 'g')
    plt.scatter(lat_true[plot_mask2], lon_true[plot_mask2], c = 'r')


def swap_streets(input_df):
    replace = {'Geary Street,Jones Street,Geary Street':'Geary Street,Jones Street,Leavenworth Street',
     'Jones Street,Sutter Street,Jones Street': 'Jones Street,Sutter Street,Cosmo Place',
     'Kearny Street,Kearny Street,Columbus Avenue': 'Kearny Street,Nottingham Place,Columbus Avenue',
     'Kearny Street,Kearny Street,Post Street': 'Kearny Street,Maiden Lane,Post Street',
     'Kearny Street,Kearny Street,Sutter Street': 'Kearny Street,Post Street,Sutter Street',
     'Larkin Street,Larkin Street,Golden Gate Avenue': 'Larkin Street,Fulton Street,Golden Gate Avenue',
     "Larkin Street,O'Farrell Street,Larkin Street": "Larkin Street,O'Farrell Street,Myrtle Street",
     'Larkin Street,Sutter Street,Larkin Street': 'Larkin Street,Sutter Street,Bush Street',
     'Mason Street,Sutter Street,Mason Street': 'Mason Street,Sutter Street,Bush Street',
     'Mission Street,11th Street,Mission Street': 'Mission Street,11th Street,12th Street',
     'Montgomery Street,Jackson Street,Montgomery Street': 'Montgomery Street,Jackson Street,Washington Street',
     'Montgomery Street,Montgomery Street,Jackson Street': 'Montgomery Street,Pacific Avenue,Jackson Street',
     'Montgomery Street,Washington Street,Montgomery Street': 'Montgomery Street,Washington Street,Clay Street',
     'Post Street,Kearny Street,Post Street': 'Post Street,Kearny Street,Montgomery Street',
     'Stockton Street,Stockton Street,Jackson Street': 'Stockton Street,Washington Street,Jackson Street',
     'Stockton Street,Stockton Tunnel,Stockton Street': 'Stockton Street,Stockton Tunnel,Clay Street',
     'Sutter Street,Powell Street,Sutter Street': 'Sutter Street,Powell Street,Mason Street',
     'Van Ness Avenue,Golden Gate Avenue,Van Ness Avenue': 'Van Ness Avenue,Golden Gate Avenue,Elm Street',
     'Van Ness Avenue,Hayes Street,Van Ness Avenue': 'Van Ness Avenue,Hayes Street,Grove Street',
     'Van Ness Avenue,McAllister Street,Van Ness Avenue': 'Van Ness Avenue,McAllister Street,Golden Gate Avenue',
     'Van Ness Avenue,Van Ness Avenue,Bush Street': 'Van Ness Avenue,Sutter Avenue,Bush Street',
     'Van Ness Avenue,Van Ness Avenue,Eddy Street': 'Van Ness Avenue,Turk Street,Eddy Street',
     'Van Ness Avenue,Van Ness Avenue,Fell Street': 'Van Ness Avenue,Oak Street,Fell Street',
     'Van Ness Avenue,Van Ness Avenue,Golden Gate Avenue': 'Van Ness Avenue,McAllister Street,Golden Gate Avenue',
     'Van Ness Avenue,Van Ness Avenue,Grove Street': 'Van Ness Avenue,Hayes Street,Grove Street',
     'Van Ness Avenue,Van Ness Avenue,Turk Street': 'Van Ness Avenue,Elm Street Avenue,Turk Street'}

    replace_fields = [k.split(',') + v.split(',') for k,v in replace.items()]
    replace_df = pd.DataFrame(replace_fields)
    replace_df.columns = ['Street', 'From', 'To', 'Clean_Street','Clean_From','Clean_To']

    proxy = input_df.merge(replace_df, how='left', on=['Street','From','To'])
    proxy.loc[proxy['Clean_Street'].isna(), 'Clean_Street'] = proxy.loc[proxy['Clean_Street'].isna(), 'Street']
    proxy.loc[proxy['Clean_From'].isna(), 'Clean_From'] = proxy.loc[proxy['Clean_From'].isna(), 'From']
    proxy.loc[proxy['Clean_To'].isna(), 'Clean_To'] = proxy.loc[proxy['Clean_To'].isna(), 'To']
    return proxy




###################################################################################################
# Model Prep
#
###################################################################################################    

def premodel_formating(train_df, split=True, test=False):
    lr_cols = ['Street.Length','holiday','dow','mon','hour','morn_night', 'From_lat','From_long','From_zip','To_lat','To_long', 'To_zip', 'nhood','zipcode','area','f_pop','m_pop','pop']
    
    if not test:
        lr_cols += ['Real.Spots','any_spot']

    train_mtx = train_df[lr_cols].copy()
    nh_enc = {v:i for i, v in enumerate(train_mtx['nhood'].unique())}
    train_mtx['nhood'] = train_mtx['nhood'].map(lambda x : nh_enc[x])
    train_mtx['zipcode'] = train_mtx['zipcode'].astype(int) 
    
    if split:
        tr, vl = train_test_split(train_mtx, test_size=0.3)        
        return tr, vl
    else:
        return train_mtx

def premodel_formating2(train_df, test_df):
    tmp = train_df.copy()
    tst = test_df.copy()

    drop_cols = ['Street', 'From', 'To', 'Time', 'min',
                'mid_address', 'id', 'Street.Length', 'Real.Spots',
                'lat_mid_y','lng_mid_y','any_spot']
    feat_cols = [col for col in train_df.columns if col not in drop_cols]

    tmp_mtx = tmp[feat_cols + ['any_spot']].copy()
    tst_mtx = tst[feat_cols].copy()
    encodings = {}
    encodings['neighborhood'] = {v:i for i, v in enumerate(tmp_mtx['neighborhood'].unique())}
    encodings['Clean_Street'] = {v:i for i, v in enumerate(tmp_mtx['Clean_Street'].unique())}
    encodings['Clean_To'] = {v:i for i, v in enumerate(tmp_mtx['Clean_To'].unique())}
    encodings['Clean_From'] = {v:i for i, v in enumerate(tmp_mtx['Clean_From'].unique())}

    for k,v in encodings.items():
        tmp_mtx[k] = tmp_mtx[k].map(v)
        tst_mtx[k] = tst_mtx[k].map(v)

    #tr, vl = train_test_split(tmp_mtx, test_size=0.25)
    # max week is 
    cut_off_week = 11
    tr = tmp_mtx[tmp_mtx['Date'].dt.week <= cut_off_week].copy()
    vl = tmp_mtx[tmp_mtx['Date'].dt.week > cut_off_week].copy()
    tr.drop(columns='Date', inplace=True)
    vl.drop(columns='Date', inplace=True)
    tst_mtx.drop(columns='Date', inplace=True)
    return tr, vl, tst_mtx, encodings




def mean_target_enc(train_df, val_df):
    tr, vl = train_df.copy(), val_df.copy()
    join_cols =['dow','hour','nhood']
    tr_mean_enc = tr.groupby(join_cols)[['any_spot','Real.Spots']].mean().reset_index()
    tr_mean_enc = tr.groupby(join_cols)[['any_spot','Real.Spots']].mean().reset_index()
    tr_mean_enc.rename(columns={'any_spot':'mean_spot_prob','Real.Spots':'mean_spot_ct'}, inplace=True)

    tr = tr.merge(tr_mean_enc, how='left', on = join_cols)
    vl = vl.merge(tr_mean_enc, how='left', on = join_cols)

    join_cols =['dow','morn_night','zipcode']
    tr_mean_enc = tr.groupby(join_cols)[['any_spot','Real.Spots']].mean().reset_index()
    tr_mean_enc = tr.groupby(join_cols)[['any_spot','Real.Spots']].mean().reset_index()
    tr_mean_enc.rename(columns={'any_spot':'z_mean_spot_prob','Real.Spots':'z_mean_spot_ct'}, inplace=True)
    
    tr = tr.merge(tr_mean_enc, how='left', on = join_cols)
    vl = vl.merge(tr_mean_enc, how='left', on = join_cols)
    
    tr.loc[tr['mean_spot_prob'].isna(),'mean_spot_prob'] = tr.loc[tr['mean_spot_prob'].isna(),'z_mean_spot_prob']
    tr.loc[tr['mean_spot_ct'].isna(), 'mean_spot_ct'] = tr.loc[tr['mean_spot_ct'].isna(),'z_mean_spot_ct']
    vl.loc[vl['mean_spot_prob'].isna(),'mean_spot_prob'] = vl.loc[vl['mean_spot_prob'].isna()]['z_mean_spot_prob']
    vl.loc[vl['mean_spot_ct'].isna(), 'mean_spot_ct'] = vl.loc[vl['mean_spot_ct'].isna(),'z_mean_spot_ct']
    print ('validation or test nulls %d, will fill with 0.4' % vl['mean_spot_prob'].isna().sum())
    vl.fillna(0.4, inplace=True)

    return tr, vl

def get_XY(tr, vl = None):
    y_tr = tr['any_spot'].values
    X_tr = tr[[col for col in tr.columns if col not in ['any_spot','Real.Spots']]].copy()
    
    if vl is not None:
        y_vl = vl['any_spot'].values
        X_vl = vl[[col for col in tr.columns if col not in ['any_spot','Real.Spots']]].copy()
        return X_tr, X_vl, y_tr, y_vl
    else:
        return X_tr, y_tr

def add_midpoint_loc_id(tmp):
    """
    assigns ids to training and test midpoint gps
    """
    mid2gps = load_midpoint_id()
    mid2gps.columns = ['lat_mid', 'lng_mid', 'loc_id']
    df = tmp.copy()
    df = df.merge(mid2gps, how='left', on=['lat_mid','lng_mid'])
    return df

###################################################################################################
# Feature Engineering
#
###################################################################################################    

def add_yelp(input_df):
    tmp = input_df.copy()
    yelp_summary_stats = pd.read_feather(REF_PATH +'yelp_summary_stats_df_by_location.feather')
    tmp = tmp.merge(yelp_summary_stats, how='left', on = 'loc_id')
    return tmp


def add_parking_vecs(input_df):
    tmp = input_df.copy()
    parkhour2vec = pd.read_feather('ref_data/parking_locid_hour2vec.feather')
    parkdow2vec = pd.read_feather('ref_data/parking_locid_dow2vec.feather')

    tmp = tmp.merge(parkhour2vec, how='left', on=['loc_id', 'hour'])
    mean_vec = parkhour2vec.groupby('loc_id').mean().reset_index()
    mean_vec.drop(columns='hour', inplace=True)

    hour_vec_cols = ['hourvec%d' % i for i in range(12) ]
    global_vec = parkhour2vec[hour_vec_cols].mean().reset_index()

    for row in mean_vec.values:
        msk = (tmp['hourvec0'].isna()) & (tmp['loc_id'] == row[0])
        for col,val in zip(hour_vec_cols, row[1:]):
            tmp.loc[msk, col] = val
            
    msk = (tmp['hourvec0'].isna())
    for col,val in zip(hour_vec_cols, global_vec[0]):
        tmp.loc[msk, col] = val
        
    tmp = tmp.merge(parkdow2vec, how='left', on=['loc_id','dow'])
    mean_vec = parkdow2vec.groupby('loc_id').mean().reset_index()
    mean_vec.drop(columns='dow', inplace=True)

    dow_vec_cols = ['dowvec%d' % i for i in range(12) ]
    global_vec = parkdow2vec[dow_vec_cols].mean().reset_index()

    for row in mean_vec.values:
        msk = (tmp['dowvec0'].isna()) & (tmp['loc_id'] == row[0])
        for col,val in zip(dow_vec_cols, row[1:]):
            tmp.loc[msk, col] = val
            
    msk = (tmp['dowvec0'].isna())
    for col,val in zip(dow_vec_cols, global_vec[0]):
        tmp.loc[msk, col] = val
    return tmp
      


@timethis
def feat_eng(df):
    """
    takes in either train or test
    returns data frame with additional columns
    """
    print('adding additional features to test and/or train ...\n')
    tmp = df.copy()
    tmp = extract_dates(tmp)
    tmp = add_tt_gps(tmp)
    tmp = tt_join_nh(tmp)
    tmp = tt_join_city_stats(tmp)
    return tmp


def extract_dates(df, field='Date'):
    """
    This function takes in a date field and pulls out month
    day of week, number day, year, etc.  into separate columns
    """
    tmp = df.copy()
    tmp['holiday'] = tmp[field].map(lambda x : 1 if x in US_HOLIDAYS else 0 )
    tmp['day']= tmp[field].dt.day
    tmp['mon']= tmp[field].dt.month
    tmp['dow']= tmp[field].dt.dayofweek
    tmp['week']= tmp[field].dt.week

    if field == 'Date':
        tmp['hour'] = tmp['Time'].map(lambda x : int(x.split(':')[0]))
        tmp['min'] = tmp['Time'].map(lambda x : int(x.split(':')[1]))
    else:
        tmp['hour'] = tmp[field].dt.hour

    tmp['morn_night'] = tmp['hour'].map(lambda x : 0 if x <10 else 1 if x <17 else 2 )
    return tmp

def add_tt_gps(df):
    tmp = df.copy()
    df_lkup = pd.read_csv(REPO_PATH+'ref_data/train_test_addr_gps.txt', delimiter='|')
    df_lkup.columns = ['id','Street','From', 'From_lat','From_long','From_zip'] 
    tmp = tmp.merge(df_lkup, how='left', on=['Street', 'From'])
    df_lkup.columns = ['id','Street','To', 'To_lat','To_long','To_zip']
    tmp = tmp.merge(df_lkup, how='left', on=['Street', 'To'])
    return tmp



def tt_join_nh(tt_df):
    """
    The goal if this function is to attach parking GPS to address
    we dont have exact keys so we are doing a closest match by Lat long
    and will assign an address to each point
    
    function is nested because it uses a semi-global variable 
    across all pools
    """
    
    print("loading parking data 1.7M")
    coord = tt_df.groupby(['To_lat','To_long'])[['day']].count().reset_index()[['To_lat','To_long']]
    coord.columns = ['lat','lon']
    coord['id'] =coord.index
          
    print("loading address data 30K")
    gps2addr = pd.read_csv( REPO_PATH + 'ref_data/clean_parking_gps2addr.txt', delimiter='|')
    keep_cols = ['nhood','road','zipcode']
    gpspts = gps2addr[['lat','lon']]
    gpspts.to_feather('gpspts')     
          
    print("calculating closest joins")
    lkup = parallelize(coord.values, closest_point)

    lkup_df = pd.DataFrame(lkup)
    lkup_df.columns = ['parking_idx','addr_idx','min_join_dist']
    
    print("merging")
    tmp = coord.merge(lkup_df, how='left', left_index=True, right_on='parking_idx')
    tmp = tmp.merge(gps2addr[keep_cols], how='left', left_on='addr_idx', right_index=True)
    join_tt_df = tt_df.merge(tmp, how='left', left_on=['To_lat','To_long'], right_on=['lat','lon'])

    print("complete")
    return join_tt_df


def tt_join_city_stats(tt_df):
    tmp = tt_df.copy()

    print('fixing neighborhood string formatting')
    tmp['neighborhood'] = tmp['nhood'].str.lower()
    tmp['neighborhood'] = tmp['neighborhood'].str.replace(' ', '')
    tmp.loc[tmp['neighborhood'] == 'russianhill-parisblockarchitecturaldistrict', 'neighborhood'] = 'russianhill'
    tmp.loc[tmp['neighborhood'] == 'libertystreethistoricdistrict', 'neighborhood'] = 'missiondistrict'
    tmp.loc[tmp['neighborhood'] == 'westsoma', 'neighborhood'] = 'southofmarket'

    print('merging in neighborhood information')
    city_stats = pd.read_csv(REPO_PATH + '/ref_data/nh_city_stats.txt',delimiter='|')
    city_stats.drop(index=[33,54], inplace=True)
    tmp = tmp.merge(city_stats, how='left', on ='neighborhood')

    return tmp

def tt_join_city_stats2clean(tt_df):
    tmp = tt_df.copy()
    city_stats = pd.read_csv(REPO_PATH + '/ref_data/nh_city_stats.txt',delimiter='|')
    city_stats.drop(index=[33,54], inplace=True)
    tmp['neighborhood'] = tmp['mid_address'].map(lambda x : x.split('|')[-1].lower().replace(' ',''))    
    tmp.loc[tmp['neighborhood']=='fillmoredistrict','neighborhood'] == 'lowerpacificheights'
    tmp = tmp.merge(city_stats[['neighborhood','pop','area','med_age']] , how = 'left', on ='neighborhood')
    return tmp

###################################################################################################
# Plotting functions
#
###################################################################################################    


def plot_dataset_overlay():
    print('loading training and test files')
    train_df = get_train()
    train_df = feat_eng(train_df)
    test_df = get_test()
    test_df = feat_eng(test_df)
    
    print('loading parking and sensor data')
    parking_df = get_parking()
    park_dist = parking_df.groupby(['lat','lon'])['datetime'].count().reset_index()
    
    for_out = process_sensor_dataframe()
    for_out_dist = for_out.groupby(['lat','lon'])['BLOCK_ID'].count().reset_index()
    
    print('plotting')
    plt.figure(figsize=(10,10))
    plt.scatter(park_dist['lat'], park_dist['lon'], c='g', alpha=0.05)
    plt.scatter(for_out_dist['lat'], for_out_dist['lon'], c='b', alpha=0.5)
    plt.scatter(train_df['To_lat'], train_df['To_long'], marker='+', c='r', alpha=.5)
    #plt.scatter(test_df['To_lat'], test_df['To_long'], marker='*', c='y', alpha=.2)
    plt.savefig(REPO_PATH + 'report_imgs/map_overlay.png')


def get_unique_coords(df, lat_field, lon_field):
    """
    Takes in a dataframe and extract unique fields for plotting.
    Uses dataframe group by to reduce the number of rows that need to
    be plotted, returns the lat and lon fields
    """
    unique = df.groupby([lat_field, lon_field]).size().reset_index()
    lat, lon= unique[lat_field].values, unique[lon_field].values
    return lat, lon


def plot_street(STREET, train_df, parking_df):
    """
    Overlays the different datasets on a 2D map for matplotlib
    """
    train_df['STREET'] = train_df['Street'].map(lambda x : x.upper())
    parking_df['STREET'] = parking_df['road'].map(lambda x : x.upper() if x is not None else x)

    kearny_t= train_df[train_df['STREET']==STREET]
    kearny_p = parking_df[parking_df['STREET']==STREET]
    
    latF, lonF = get_unique_coords(kearny_t, 'From_lat', 'From_long')
    latT, lonT = get_unique_coords(kearny_t, 'To_lat', 'To_long')
    latp, lonp = get_unique_coords(kearny_p, 'lat', 'lon')
    
    
    
    plt.figure(figsize=(7,7))
    plt.title('STREET : %s' % STREET)
    plt.scatter(latp, lonp, c='g', alpha = 0.5)
    plt.scatter(latF, lonF, c='b', alpha=0.5)
    plt.scatter(latT, lonT, c='black', alpha = 0.5)


###################################################################################################
# Parking Related Functions
#
###################################################################################################    


@timethis
def get_parking(force=False):
    """
    imports parking records, adds in the addresses
    returns as supplementary data
    """
    df_parking = pd.read_csv(DATA_PATH + 'parkingrecords.csv')
    df_parking = parking_join_addr(df_parking, force=force) 
    return df_parking

    
def parking_join_addr(parking_df, force=False):
    """
    The goal if this function is to attach parking GPS to address
    we dont have exact keys so we are doing a closest match by Lat long
    and will assign an address to each point
    
    function is nested because it uses a semi-global variable 
    across all pools
    """
    save_path = DATA_PATH + 'P_parking_clean.feather'

    if (os.path.isfile(save_path)) and (not force):
        print('loading cached copy')
        join_parking_df = pd.read_feather(save_path)
        return join_parking_df
    else:
        print("loading parking data 1.7M")
        park_dist = parking_df.groupby(['lat','lon'])[['datetime']].count().reset_index()[['lat','lon']]
        park_dist['id'] =park_dist.index
              
        print("loading address data 30K")
        gps2addr = pd.read_csv(REPO_PATH + 'ref_data/clean_parking_gps2addr.txt', delimiter='|')
        keep_cols = ['full_addr','jlat','jlon','nhood','road','zipcode']
        gpspts = gps2addr[['lat','lon']]
        gpspts.to_feather('gpspts')     
              
        print("calculating closest joins")
        lkup = parallelize(park_dist.values, closest_point)

        lkup_df = pd.DataFrame(lkup)
        lkup_df.columns = ['parking_idx','addr_idx','min_join_dist']
        
        print("merging")
        tmp = park_dist.merge(lkup_df, how='left', left_index=True, right_on='parking_idx')
        tmp = tmp.merge(gps2addr[keep_cols], how='left', left_on='addr_idx', right_index=True)
        join_parking_df = parking_df.merge(tmp, how='left', on=['lat','lon'])

        join_parking_df.to_feather(save_path)
        print("complete")
        return join_parking_df



def add_street_sections(train_df, test_df=None, force=False):
    
    train_streets = list(train_df['Street'].unique())
    train_streets = [s.upper() for s in train_streets]
    
    parking_df = get_parking()
    parking_street = list(parking_df['road'].unique())
    parking_street = [s.upper() for s in parking_street if s is not None]
    
    
    print('Identifying key streets...')
    # upper case streets for joining
    train_df['STREET'] = train_df['Street'].map(lambda x : x.upper())
    parking_df['STREET'] = parking_df['road'].map(lambda x : x.upper() if x is not None else x)

    print('Filtering down the parking data ...')
    # down filter to only training streets
    parking_sm_df = parking_df[parking_df['STREET'].isin(train_streets)]
    parking_sm_df.shape
    
    VERT_STREET = ['MISSION STREET', 'PINE STREET', '2ND STREET', 'BRYANT STREET', 
               'POST STREET', 'WASHINGTON STREET', 'JACKSON STREET', 'BUSH STREET', 
               'GEARY STREET', 'GROVE STREET', 'SUTTER STREET', 'COLUMBUS AVENUNE']
    
    print('measuring parking coords within FROM TO bounds ...')
    dist_parking = parking_sm_df.groupby(['lat','lon','STREET']).size().reset_index()
    dist_tr = train_df.groupby(['From_lat','From_long','To_lat','To_long','STREET']).size().reset_index()
    dist_tr['STR_SEC'] = dist_tr.index

    for latf, lonf, latt,lont, street, ct, id_ in dist_tr.values:
        min_lat = min(latf, latt)
        max_lat = max(latf, latt)
        min_lon = min(lonf, lont)
        max_lon = max(lonf, lont)

        if street in VERT_STREET:
            dist_parking.loc[(dist_parking['STREET'] == street) & (dist_parking['lon'] < max_lon) & (dist_parking['lon'] > min_lon),"STR_SEC" ] = id_ 
        else:
            dist_parking.loc[(dist_parking['STREET'] == street) & (dist_parking['lat'] < max_lat) & (dist_parking['lat'] > min_lat) ,"STR_SEC" ] = id_ 

    print('Adding street section ids to train and parking ...')
    rejoined_parking = parking_sm_df.merge(dist_parking[['lat','lon','STR_SEC']], how ='left', on=['lat','lon'])
    rejoined_parking = rejoined_parking[rejoined_parking['STR_SEC'].isna()!=True].copy()
    
    rejoined_parking['CAL_DATE'] = rejoined_parking['datetime'].map(lambda x : dt.datetime.fromtimestamp(x/1000.0))
    rejoined_parking = extract_dates(rejoined_parking, 'CAL_DATE')

    join_cols = ['From_lat','From_long','To_lat','To_long']
    train_df = train_df.merge(dist_tr[join_cols+['STR_SEC']], how ='left', on =join_cols)
    
    print('Complete')
    if test_df is not None:
        test_df = test_df.merge(dist_tr[join_cols+['STR_SEC']], how ='left', on =join_cols)
        return rejoined_parking, train_df, test_df
    else:
        return rejoined_parking, train_df


###################################################################################################
# Sensor Data
#
###################################################################################################    

# for sensor data
def reorder_street_block(strr):
    if strr is not None:
        num = strr.split(' ')[-1]
        rest = ' '.join(strr.split(' ')[:-1])
        if rest[0] == '0':
            rest = rest[1:]
        return('%s %s' % (num, rest))
    else:
        return strr


@timethis
def process_sensor_dataframe(force=False):
    save_path = DATA_PATH + 'P_sensor_clean.feather'
    
    if (os.path.isfile(save_path)) and (force==False):
        print('loading cached copy')
        join_sn_df = pd.read_feather(save_path)
        return join_sn_df
    
    else:
    
        print('loading feathered dataframe')
        sn_df = pd.read_feather(DATA_PATH+'P_sensor.feather')

        print('converting to date time')
        dt_str = sn_df['START_TIME_DT'].unique()
        lkup = {str_v : pd.to_datetime(str_v) for str_v in dt_str}

        # total run time is :

        sn_df['START_TIME_DT'] = sn_df['START_TIME_DT'].map(lambda x : lkup[x])
        sn_df['tot_spaces'] = sn_df['TOTAL_TIME'] / 3600.
        sn_df['occ_spaces'] = sn_df['TOTAL_OCCUPIED_TIME'] / 3600.
        sn_df['vac_spaces'] = sn_df['TOTAL_VACANT_TIME'] / 3600.
        sn_df['unk_spaces'] = sn_df['TOTAL_UNKNOWN_TIME'] / 3600.
        sn_df['occ_spaces_pct'] = sn_df['occ_spaces']/ sn_df['tot_spaces']
        sn_df['vac_spaces_pct'] = sn_df['vac_spaces'] / sn_df['tot_spaces']
        sn_df['unk_spaces_pct'] = sn_df['unk_spaces'] / sn_df['tot_spaces']
        sn_df['day'] = sn_df['START_TIME_DT'].dt.day
        sn_df['mon'] = sn_df['START_TIME_DT'].dt.month
        sn_df['dow'] = sn_df['START_TIME_DT'].dt.dayofweek
        sn_df['hour'] = sn_df['START_TIME_DT'].dt.hour

        keep_cols = ['BLOCK_ID','STREET_NAME', 'BLOCK_NUM', 'STREET_BLOCK', 
                     'PM_DISTRICT_NAME', 'RATE','START_TIME_DT', 'tot_spaces','occ_spaces',
                     'vac_spaces','unk_spaces', 'vac_spaces_pct','unk_spaces_pct', 'occ_spaces_pct',
                     'day','mon','dow','hour']

        sn_df = sn_df[keep_cols].copy()

        print('starting gps merge')

        # merging in the GPS info    
        gps_df = pd.read_csv(REPO_PATH + 'ref_data/sensor_2_gps.txt', delimiter='|', header=None)
        gps_df.columns = ['id', 'STREET_B', 'lat','lon','zip']
        gps_df.drop(columns='id', inplace=True)

        #prep index
        sn_df['STREET_B'] = sn_df['STREET_BLOCK'].map(lambda x : reorder_street_block(x))
        join_sn_df = sn_df.merge(gps_df, how='left', on='STREET_B')

        print('complete... saving cached copy')

        join_sn_df.to_feather(save_path)
        return join_sn_df
    
def sensor_make_avg_tables(df):
    """
    Because the span of dates is so different, we will have to make average tables
    of our sensor data to attach it to our main training data
    will take in stats from 2011-2013 and make an average table
    based on MONTH, DAY OF WEEK, and HOUR
    """
    tmp = df.copy()
    group_feat=['BLOCK_ID','STREET_NAME','BLOCK_NUM','PM_DISTRICT_NAME','mon','dow','hour','lat','lon','zip']
    metric_feat =['tot_spaces','occ_spaces','vac_spaces','unk_spaces', \
                      'occ_spaces_pct','vac_spaces_pct','unk_spaces_pct'] 
    avg_tbl = tmp.groupby(group_feat)[metric_feat].mean().reset_index()
    avg_tbl.columns = group_feat + [col +'_avg' for col in metric_feat]
    avg_tbl['STREET'] = avg_tbl['STREET_NAME'].map(lambda x : x[1:] if x[0]=='0' else x)
    
    for col in ['mon','dow','hour']:
        avg_tbl[col] = avg_tbl[col].astype(int)
        
    return avg_tbl
    

def parallelize(inputdata, func):
    """
    generic parallelizing class
    """    
    cores = cpu_count() 
    
    data_split = np.array_split(inputdata, cores)
    pool = Pool(cores)
    out = np.concatenate(pool.map(func, data_split), axis=0)
    pool.close()
    pool.join()
    return out


def closest_point(park_dist):
    """
    point by point traverses a list and
    pulls out the index of the closest point 
    from a reference vector
    """
    gpspts = pd.read_feather('gpspts')
    output = np.zeros((park_dist.shape[0], 3), dtype=int)
    for i, point in enumerate(park_dist):
        x,y, id_ = point
        sq_dist = np.sqrt(np.power(gpspts.iloc[:,0]-x,2) + np.power(gpspts.iloc[:,1]-y,2))
        dist = np.log(sq_dist + 0.0000001)
        output[i,:] = (id_,dist.idxmin(),np.min(dist))
    return output


    