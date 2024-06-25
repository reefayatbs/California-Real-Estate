import json
import pickle
import numpy as np

__city = None
__data_columns = None
__model = None

def get_estimated_price(city, house_size, bed, bath):
    try:
        city_index = __data_columns.index(city.lower())
    except:
        city_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = bed
    x[1] = bath
    x[2] = house_size
    if city_index >= 0:
        x[city_index] = 1

    return round(__model.predict([x])[0], 2)

def get_city_names():
    return __city

def load_artifacts():
    print("loading saved artifacts...start")
    global __data_columns
    global __city

    with open("./artifacts/columns.json", "r") as f:
        __data_columns = json.load(f)["data_columns"]
        __city = __data_columns[3:]

    global __model
    with open("./artifacts/California_RealEstate_model.pickle", 'rb') as f:
        __model = pickle.load(f)

    print("loading saved artifacts...done")

if __name__ == "__main__":
    load_artifacts()
    print(get_city_names())
    print(get_estimated_price('Los Angeles', 1000, 2, 2))
    print(get_estimated_price('Los Angeles', 1000, 3, 3))
    print(get_estimated_price('New York', 1000, 2, 2))
