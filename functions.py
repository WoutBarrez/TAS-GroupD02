import pandas as pd
def read(drone_name):
    values_gps = 'gps/GPS/'+drone_name+'_GPS.csv' #reading file for the gps
    gps_read = pd.read_csv(values_gps) #read the values using pandas
    return gps_read #returns name index and the pandas read of gps values
def avoid_errors(gps_read):
    df = pd.DataFrame(gps_read)

    # Specify the specific values you want to exclude
    values_to_exclude = [0, None]

    # Create a boolean mask for rows containing specific values or NaN
    mask = ~(df.isin(values_to_exclude) | df.isna()).any(axis=1)

    # Use the mask to filter the DataFrame and keep rows without specific values
    df_filtered = df[mask]

    print(df_filtered)

def rotate_positions()