
import requests
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os,sys

sys.path.append(os.path.abspath(os.path.join('C:\Work\WORK_PACKAGE\Demand_forecasting\github\DeepAR-pytorch\My_model\2_freq_nbinom_LSTM\1_cluster_demand_prediction\data\weather_data')))


########### API calls #############

# lat = 1.352566
# lon = 103.944569
# lat = str(lat)
# lon = str(lon)
# t = 1632412800 - 608400
# et = 1640257200
# cluster_list = []
# for i in range(1,14):
#     t = t + 608400
#     start = str(t)
#     cluster = "tampines_175_" + str(i)
#     cluster_list.append(cluster)
#     response = response = requests.get("https://history.openweathermap.org/data/2.5/history/city?lat=" + lat + "&lon=" + lon + "&type=hour&start=" + start + "&end=1640257200&appid=09bb9a014923359732cd5ce17a1827c4")
#     with open(cluster+'.json',"w") as f:
#         json.dump(response.json(),f)

########### API calls #############





########### SAVE region weather data #############

cluster_list = ["tampines_175_1","tampines_175_2","tampines_175_3","tampines_175_4","tampines_175_5","tampines_175_6","tampines_175_7", "tampines_175_8","tampines_175_9","tampines_175_10","tampines_175_11","tampines_175_12","tampines_175_13"]
date_time = np.array([])
temp_clstr_175 = np.array([])
hum_clstr_175 = np.array([])
wind_clstr_175 = np.array([])
wea_clstr_175 = np.array([])
wea_desc_clstr_175 = np.array([])

for c in cluster_list:
    with open(c+".json",'r') as f:
        data = json.load(f)
        for i in range(len(data["list"])):
            date_time = np.append(date_time, data["list"][i]["dt"])
            temp_clstr_175 = np.append(temp_clstr_175, data["list"][i]["main"]["temp"])
            hum_clstr_175 = np.append(hum_clstr_175, data["list"][i]["main"]["humidity"])
            wind_clstr_175 = np.append(wind_clstr_175, data["list"][i]["wind"]["speed"])
            wea_clstr_175 = np.append(wea_clstr_175, data["list"][i]["weather"][0]["main"])
            wea_desc_clstr_175 = np.append(wea_desc_clstr_175, data["list"][i]["weather"][0]["description"])


#plt.plot(temp_clstr_175)
plt.plot(wind_clstr_175)
plt.title("cluster 175/wind speed")
plt.show()

df = pd.DataFrame()
df["temp_clstr_175"] = temp_clstr_175
df["hum_clstr_175"] = hum_clstr_175
df["wind_clstr_175"] = wind_clstr_175
df["wea_clstr_175"] = wea_clstr_175
df["wea_desc_clstr_175"] = wea_desc_clstr_175
df.to_csv("tampines_clstr_175_weather.csv")

########### SAVE region weather data #############

















