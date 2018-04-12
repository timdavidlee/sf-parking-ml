### These are report notes

- Tree limit , with training data:
- log_2 (1100) ~ 10 
- Don't go beyond tree depth 10 (will over fit)

Training Set

- 1,100 rows from 2014
- 3 Three months covered
- Jan - Mar

Test set (wide array of dates)

- 726 
- March 2014
- Nov 2016

Parking Dataset

- 1,761,328
- 28,159 unique locations

Sensor Dataset
- from 2011 - 2013
- (7,902,291, 33) rows
- 407 unique block locations 

### B. Lookups

#### B1. Parking data has GPS coords --> pulled streets + address + zip
```
ref_data/clean_parking_gps2addr.txt
	
	17, Leland Avenue, Visitacion Valley, SF, California, 94134, United States of America|37.7112803333333|-122.404178|37.711310|-122.404174|Visitacion Valley|Leland Avenue|94134
```

#### B2. for cleaning up names, have a simple lookup for SF neighborhoods
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

```
ref_data/nh_city_stats.txt
	|_id|apt2_avg_value|apt2_pct|apt4_avg_value|apt4_pct|area|f_pop|house_avg_value|house_pct|m_pop|med_age|neighborhood|pop|twn_avg_value|twn_pct
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


### C EDA and other Statistics

- 158 unique combos From + To combinations (both train and test)
- ~28k unique GPS combinations from the parking records


![](https://snag.gy/iflyKt.jpg)

### Day of Week / Morn / Afternoon / Night
![](https://snag.gy/VCHWe8.jpg)

### Hour of the day

![](https://snag.gy/YGQFkU.jpg)


### Sample plots of streets and parking 

![](https://snag.gy/nXeZpD.jpg)

![](https://snag.gy/JXf2Td.jpg)

![](https://snag.gy/WLcFps.jpg)

![](https://snag.gy/UjDXkL.jpg)