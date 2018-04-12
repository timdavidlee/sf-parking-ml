# ML630 - Internal Kaggle Competition

### Welcome to ChengCheng and Tim's Kaggle Resources:

<img src='https://snag.gy/0qp6dV.jpg' style="width:200px;" />

[https://www.kaggle.com/c/parking-in-sf](https://www.kaggle.com/c/parking-in-sf)

## Goal: Predict Parking Availability

Given:

- Street
- Time of Day
- Day of Week
- Street Intersection

Predict:

- Is there parking available? Y / N 

## Datasets

<table>
<tr>
	<td> Training Set </td>
	<td>
	- 1,100 rows from 2014 <br>
	- 3 Three months covered <br>
	- Jan - Mar <br>
	</td>
</tr>
<tr>
	<td>Test set (wide array of dates)</td>
	<td>
	- 726  <br>
	- March 2014 <br>
	- Nov 2016 <br>
	</td>
</tr>
<tr>
<td>
Parking Dataset
</td>
<td>

- 1,761,328 <br>
- 28,159 unique locations <br>

</td>
</tr>
<tr>
<td>
Sensor Dataset
</td>
<td>
- from 2011 - 2013 <br>
- (7,902,291, 33) rows <br>
- 407 unique block locations <br> 
</td>
</tr>

</table>





### A. Navigating this repo
```
repo
|--eda/				| Has exploration notebooks for the different datasets
|						| Also has some prototyping work for combining datasets
|						|
|--ref_data/			| has all extra scraped data (that will fit on github)
|--submissions/		| reference submissions for Tim and ChengCheng
|- common.py			| contains all key common functions used for study
```

#### Custom Functions in `common.py`
```
common.py
--get_train()
--get_test()
--plot_pred_coord(y_true, lat_true, lon_true, y_pred)
--premodel_formating(train_df, split=True, test=False)
--mean_target_enc(train_df, val_df)
--get_XY(tr, vl = None)
--feat_eng(df)
--extract_dates(df, field='Date')
--add_tt_gps(df)
--tt_join_nh(tt_df)
--tt_join_city_stats(tt_df)
--plot_dataset_overlay()
--get_unique_coords(df, lat_field, lon_field)
--plot_street(STREET, train_df, parking_df)
--get_parking(force=False)
--parking_join_addr(parking_df, force=False)
--add_street_sections(train_df, test_df=None, force=False)
--reorder_street_block(strr)
--process_sensor_dataframe(force=False)
--sensor_make_avg_tables(df)
--parallelize(inputdata, func)
--closest_point(park_dist)
```

--- 
## Current notes

Still optimizing models found in the base directory. 

#### Observations
1. training data is really small
2. test data is from a different time period
3. very easy to overfit the training data
4. The sensor data ( dark blue in the initial picture) only has minimal overlap
5. The parking data has better coverage, but still missing in some streets
6. The training data has intersections of KEARNY and KEARNY, not sure what to make of this

### Parking data on the same street
<img src='https://snag.gy/6FihgX.jpg' style='width:500px' />

### Parking data truncated by training blocks
<img src='https://snag.gy/YimcrG.jpg' style='width:500px' />

<img src='https://snag.gy/PRCjG1.jpg' style='width:400px'/>

#### Sample of finding parking between street points
<img src='https://snag.gy/KJVx92.jpg' />

#### Sample of parking 'around the corner'
<img src='https://snag.gy/VbeSAZ.jpg' />


---
### B. Supplementary Data

#### B1. Parking data has GPS coords --> pulled streets + address + zip
This is currently being explored to be joined to augment the training data. Though initial analysis shows only about 30% coverage of the training data. Further analysis is needed.

```
ref_data/clean_parking_gps2addr.txt
	
	17, Leland Avenue, Visitacion Valley, SF, California, 94134, United States of America|
	37.7112803333333|-122.404178|37.711310|-122.404174|Visitacion Valley|Leland Avenue|94134
```

#### B2. for cleaning up names, have a simple lookup for SF neighborhoods

Since the names come in compound words and have caps + lowercase, this lookup is used to standardize all the names

```
ref_data/neighname2condensed.txt

	["Alamo Square":"alamosquare"
	,"Anza Vista":"anzavista"
	,"Aquatic Park":"aquaticpark"
	,"Baja Noe":"bajanoe"
	,"Balboa Park":"balboapark"
	,"Balboa Terrace":"balboaterrace"
	,"Bayview District":"bayviewdistrict"
	,"Bayview Heights":"bayviewheights"
	,"Bernal Heights":"bernalheights"
```

#### B3. General city statistics

[http://www.city-data.com/nbmaps/neigh-San-Francisco-California.html](http://www.city-data.com/nbmaps/neigh-San-Francisco-California.html)

For the possibility of target encoding or understanding parking by neighborhood, these summary statistics were scraped from the city data website.

```
ref_data/nh_city_stats.txt
	|_id|apt2_avg_value|apt2_pct|apt4_avg_value|apt4_pct|area|f_pop|
			house_avg_value|house_pct|m_pop|med_age|neighborhood|pop|twn_avg_value|twn_pct
	0|0.0|1215016.0|12.3|916341.0|26.1|0.144|38.8|2122937.0|4.5|35.7|38.8|alamosquare|5903.0|1041250.0|2.6
```

#### B4. Sensor has STREETS --> GPS data

```
ref_data/sensor_2_gps.txt

	0|0 1ST ST|37.7910492|-122.39916449999998|94111
	1|200 1ST ST|37.788551399999996|-122.39602530000002|94105
	2|300 1ST ST|37.787330100000005|-122.39451830000002|94105
```

#### B5. Zip code to Neighborhood lookup
```
ref_data/sf_nhood_zip_lkup.txt

	{
	94102 : "Hayes Valley"
	,94103 : "SoMa"
	,94104 : "Financial District"
	,94105 : "Embarcadero South"
	,94107 : "Portrero Hill"
	,94108 : "Chinatown"
	,94109 : "San Francisco"
	,94109 : "Nob Hill"
	,94109 : "Russian Hill"
```

#### B6. From the train and Test, street - GPS

```
ref_data/train_test_addr_gps.txt

	id|Street|From_To|lat|long|zip
	0|Taylor Street|Geary Street|37.786946500000006|-122.41154790000002|94102
	1|Van Ness Avenue|Turk Street|37.7819764|-122.4205794|94102
	2|Jones Street|Sutter Street|37.7886086|-122.41356480000002|94109
	3|Van Ness Avenue|Alice B. Toklas Place|37.785244399999996|-122.4212351|94109
	4|Bush Street|Taylor Street|37.7897577|-122.412102|94108
	5|Mission Street|25th Street|37.7506482|-122.41831589999998|94110
	6|23rd Street|Bartlett Street|37.7537794|-122.41972909999998|94110
	7|Battery Street|Halleck Street|37.7937021|-122.40008469999998|94111
```

--- 

### C EDA and other Statistics

- 158 unique combos From + To combinations (both train and test)
- ~28k unique GPS combinations from the parking records

<img src='https://snag.gy/iflyKt.jpg' style='width:400px' />


### Day of Week / Morn / Afternoon / Night
<img src='https://snag.gy/VCHWe8.jpg' style='width:400px' />

### Hour of the day
<img src='https://snag.gy/YGQFkU.jpg' style='width:400px' />


### Sample plots of streets and parking 
<img src='https://snag.gy/nXeZpD.jpg' style='width:400px' />

<img src='https://snag.gy/JXf2Td.jpg' style='width:400px' />

<img src='https://snag.gy/WLcFps.jpg' style='width:400px' />

<img src='https://snag.gy/UjDXkL.jpg' style='width:400px' />