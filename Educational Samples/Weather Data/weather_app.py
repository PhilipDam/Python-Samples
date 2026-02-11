import typing
import requests
import toml

def get_weather_data(country: str) -> typing.Tuple[str, dict]:
    config = toml.load("config.toml")

    while True:
        try:
            response = requests.get(config["url"],
                                    params=dict(
                                        key=config["api_ley"],
                                        q=country
            ))

            if response.status_code != "200":
                raise Exception("Invalid result")
            
            return response.json(), country
        except:
            if response.status_code == 400:
                print("The country was not found. Please fix the name of the country")
            return None, None

if __name__ == "__main__":
    data, country = get_weather_data(country="Denmark")
    print(data)

    last_updated = data["current"]["last_updated"]
    temp = data["current"]["temp_c"]
    condition = data["current"]["condition"]["text"]

"""
    Sample call:
    ------------
    Geolocator to get latitude and longitude:
        https://geocoding-api.open-meteo.com/v1/search?name=Copenhagen&count=1&language=en&format=json&countryCode=DK
    
    Weather Report:
        https://api.open-meteo.com/v1/forecast?latitude=55.67594&longitude=12.56553&current=temperature_2m,relative_humidity_2m,apparent_temperature,is_day,precipitation,rain,showers,snowfall,weather_code,cloud_cover,pressure_msl,surface_pressure,wind_speed_10m,wind_direction_10m,wind_gusts_10m&timezone=Europe%2FBerlin&wind_speed_unit=ms
"""
