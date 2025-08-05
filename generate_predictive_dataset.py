import os
import requests
import csv
import pandas as pd
import io
from dotenv import load_dotenv
from datetime import date, timedelta, datetime
import json
from xert_api_client import XertApiClient

load_dotenv()

api_key = os.getenv("INTERVALS_API_KEY")
athlete_id = os.getenv("INTERVALS_ATHLETE_ID")
XERT_CLIENT_ID = os.getenv("XERT_CLIENT_ID")
XERT_CLIENT_SECRET = os.getenv("XERT_CLIENT_SECRET")
xert_client = XertApiClient(XERT_CLIENT_ID, XERT_CLIENT_SECRET)

# Helper function to get pre-race training metrics
def get_pre_race_training_metrics(athlete_id, api_key, window_end_date, days_prior, taper_days=0):
    actual_end_date = window_end_date - timedelta(days=taper_days)

    start_date = actual_end_date - timedelta(days=days_prior)
    activities_url = f"https://intervals.icu/api/v1/athlete/{athlete_id}/activities?oldest={start_date.isoformat()}&newest={actual_end_date.isoformat()}"

    pre_race_activities = []
    try:
        response = requests.get(activities_url, auth=("API_KEY", api_key))
        response.raise_for_status()
        pre_race_activities = response.json()
        print(f"Debug: Fetched {len(pre_race_activities)} pre-race activities for {window_end_date} ({days_prior} days prior).")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching pre-race activities for {window_end_date} ({days_prior} days prior): {e}")
        return None, None, None

    total_time_above_38c = 0 # in seconds
    ef_values = []
    total_moving_time = 0 # in seconds

    for activity in pre_race_activities:
        # Only consider training activities, not races
        if activity.get('race') is not True:
            # Time spent above 38°C body temperature
            core_temp_stream_found = False
            # Attempt to fetch core_temperature stream directly
            core_temp_stream_url = f"https://intervals.icu/api/v1/activity/{activity['id']}/stream?stream=core_temperature"
            try:
                core_temp_response = requests.get(core_temp_stream_url, auth=("API_KEY", api_key))
                core_temp_response.raise_for_status()
                core_temp_data = core_temp_response.json()

                if core_temp_data and 'data' in core_temp_data:
                    core_temp_values = [float(val) for val in core_temp_data['data'] if val is not None]
                    if core_temp_values:
                        time_over_38 = sum(1 for val in core_temp_values if val > 38.0)
                        total_time_above_38c += time_over_38
                        print(f"Debug: Activity {activity['id']} had {time_over_38}s above 38C from stream 'core_temperature'.")
                        core_temp_stream_found = True
                    else:
                        print(f"Debug: No data or empty data for stream 'core_temperature' for activity {activity['id']}")
                else:
                    print(f"Debug: No data or empty data for stream 'core_temperature' for activity {activity['id']}")
            except requests.exceptions.RequestException as e:
                print(f"Debug: Error fetching stream 'core_temperature' for activity {activity['id']}: {e}")
                pass # Continue if error

            if not core_temp_stream_found:
                print(f"Debug: No core temperature data found for activity {activity['id']} in any checked stream.")

            # Average cardiovascular performance index (EF)
            if activity.get('icu_efficiency_factor') is not None:
                ef_values.append(activity['icu_efficiency_factor'])

            # Total moving time for training activities
            if activity.get('moving_time') is not None:
                total_moving_time += activity['moving_time']

    avg_ef = sum(ef_values) / len(ef_values) if ef_values else None

    return total_time_above_38c, avg_ef, total_moving_time

# Helper function to get Xert activity metrics
def get_xert_activity_metrics(xert_client, activity_id):
    xert_activity_details = xert_client.get_activity_details(activity_id)
    if xert_activity_details:
        return {
            'Xert XSS': xert_activity_details.get('xss'),
            'Xert XEP': xert_activity_details.get('xep'),
            'Xert Focus': xert_activity_details.get('focus'),
            'Xert Specificity': xert_activity_details.get('specificity'),
            'Xert Activity Type': xert_activity_details.get('type'),
            'Xert Duration': xert_activity_details.get('duration'),
            'Xert Avg Power': xert_activity_details.get('avg_power'),
            'Xert Avg HR': xert_activity_details.get('avg_heart_rate'),
            'Xert Avg Cadence': xert_activity_details.get('avg_cadence'),
            'Xert Avg Speed': xert_activity_details.get('avg_speed'),
            'Xert Max Power': xert_activity_details.get('max_power'),
            'Xert Max HR': xert_activity_details.get('max_heart_rate'),
            'Xert Max Speed': xert_activity_details.get('max_speed'),
        }
    return {}

# --- Data Collection and Processing ---

# Define a very wide date range to fetch all activities
oldest_date_activities = date(2000, 1, 1)
today = date.today()

# Construct the API URL for all activities with a wide date range
activities_url = f"https://intervals.icu/api/v1/athlete/{athlete_id}/activities?oldest={oldest_date_activities.isoformat()}&newest={today.isoformat()}"

try:
    # Make the API request for activities
    print("Fetching all activities...")
    response = requests.get(activities_url, auth=("API_KEY", api_key))
    response.raise_for_status()  # Raise an exception for bad status codes

    activities_data = response.json()

    # --- Debugging: Print raw JSON for specific activities ---
    debug_activity_ids = ['i49131928', 'i1419320', 'i23157401'] # Top 3 races by Performance Score
    for debug_activity_id in debug_activity_ids:
        for activity in activities_data:
            if activity.get('id') == debug_activity_id:
                print(f"\n--- Raw JSON for Activity ID: {debug_activity_id} ---")
                print(json.dumps(activity, indent=2))
                break
    # --- End Debugging ---

    print(f"Processing {len(activities_data)} activities. Filtering for races and generating predictive dataset...")
    predictive_dataset_rows = []
    prediction_horizons = [7, 14, 21, 28, 35, 42] # Days before the race for prediction

    for activity in sorted(activities_data, key=lambda x: x['id']):
        if activity.get('race') is True:
            race_date_str = activity['start_date_local'].split('T')[0]
            race_date = datetime.strptime(race_date_str, '%Y-%m-%d').date()

            # Calculate Performance Score for the race (this is the target variable)
            # Normalize speed and kilojoules for scoring
            # Avoid division by zero if all values are the same
            avg_speed_kmh = round(activity.get('average_speed', 0) * 3.6, 2)
            icu_efficiency_factor = activity.get('icu_efficiency_factor')
            icu_weighted_avg_watts = activity.get('icu_weighted_avg_watts')

            # Fetch weight for Power/Weight calculation (with fallback to nearest)
            weight_on_race_day = None
            # Try to get weight on race day
            wellness_url_race_day = f"https://intervals.icu/api/v1/athlete/{athlete_id}/wellness?oldest={race_date.isoformat()}&newest={race_date.isoformat()}"
            try:
                wellness_response_race_day = requests.get(wellness_url_race_day, auth=("API_KEY", api_key))
                wellness_response_race_day.raise_for_status()
                wellness_data_race_day = wellness_response_race_day.json()
                if wellness_data_race_day and wellness_data_race_day[0].get('weight') is not None:
                    weight_on_race_day = wellness_data_race_day[0]['weight']
            except requests.exceptions.RequestException:
                pass # No wellness data for this day or error fetching it

            # If no weight on race day, try to find the nearest within +/- 7 days
            if weight_on_race_day is None:
                search_start_date = race_date - timedelta(days=30)
                search_end_date = race_date + timedelta(days=30)
                wellness_url_nearby = f"https://intervals.icu/api/v1/athlete/{athlete_id}/wellness?oldest={search_start_date.isoformat()}&newest={search_end_date.isoformat()}"
                try:
                    wellness_response_nearby = requests.get(wellness_url_nearby, auth=("API_KEY", api_key))
                    wellness_response_nearby.raise_for_status()
                    wellness_data_nearby = wellness_response_nearby.json()

                    if wellness_data_nearby:
                        # Sort by proximity to race_date
                        wellness_data_nearby.sort(key=lambda x: abs((datetime.strptime(x['id'], '%Y-%m-%d').date() - race_date).days))
                        for entry in wellness_data_nearby:
                            if entry.get('weight') is not None:
                                weight_on_race_day = entry['weight']
                                break
                except requests.exceptions.RequestException:
                    pass # No nearby wellness data or error fetching it

            power_weight_w_kg = None
            try:
                if icu_weighted_avg_watts is not None and weight_on_race_day is not None and weight_on_race_day > 0:
                    power_weight_w_kg = round(icu_weighted_avg_watts / weight_on_race_day, 2)
                else:
                    power_weight_w_kg = None
            except TypeError:
                power_weight_w_kg = None

            # Store the raw values for later Performance Score calculation
            race_raw_metrics = {
                'race_id': activity['id'],
                'race_date': race_date_str,
                'race_name': activity['name'],
                'race_type': activity['type'],
                'race_avg_speed_kmh': avg_speed_kmh,
                'race_efficiency_factor': icu_efficiency_factor,
                'race_power_weight_w_kg': power_weight_w_kg,
                'race_kilojoules': activity.get('icu_joules'),
                'race_moving_time_minutes': round(activity.get('moving_time', 0) / 60, 2),
                'race_max_hr': activity.get('max_heartrate'),
                'Variability': activity.get('icu_variability_index'),
                'Power/HR': activity.get('icu_power_hr'),
                'Carbohydrates Consumed (g)': activity.get('carbs_used'),
                'Max Power (W)': activity.get('max_power'),
                'Average Cadence (rpm)': activity.get('average_cadence'),
                'Total Ascent (m)': activity.get('total_elevation_gain'),
                'Total Descent (m)': activity.get('total_elevation_loss'),
                'Normalized Power (W)': activity.get('icu_weighted_avg_watts'),
                'Training Stress Score': activity.get('icu_training_load'),
                'Intensity Factor': activity.get('icu_intensity'),
                'Average Temp (C)': activity.get('average_temp'),
                'Max Temp (C)': activity.get('max_temp'),
                'Min Temp (C)': activity.get('min_temp'),
                'Incident': None, # Placeholder for future incident data
                'Kilojoules/Hour': round(activity.get('icu_joules', 0) / (activity.get('moving_time', 1) / 3600), 2) if activity.get('moving_time', 1) > 0 else 0,
                'Peak Core Temp (C)': None, # Will be fetched later if available
                'Time Over 38C (s)': None, # Will be fetched later if available
            }
            # Fetch Xert activity metrics for the race itself
            xert_metrics_race = get_xert_activity_metrics(xert_client, activity['id'])
            for k, v in xert_metrics_race.items():
                race_raw_metrics[f'race_{k.replace(" ", "_").lower()}'] = v

            # --- Fetch Core Temperature Data for the race itself ---
            core_temp_stream_url = f"https://intervals.icu/api/v1/activity/{activity['id']}/stream?stream=core_temperature"
            try:
                core_temp_response = requests.get(core_temp_stream_url, auth=("API_KEY", api_key))
                core_temp_response.raise_for_status()
                core_temp_data = core_temp_response.json()

                if core_temp_data and 'data' in core_temp_data:
                    core_temp_values = [float(val) for val in core_temp_data['data'] if val is not None]
                    if core_temp_values:
                        race_raw_metrics['Peak Core Temp (C)'] = max(core_temp_values)
                        time_over_38 = sum(1 for val in core_temp_values if val > 38.0)
                        race_raw_metrics['Time Over 38C (s)'] = time_over_38
                        print(f"Debug: Activity {activity['id']} had {time_over_38}s above 38C from stream 'core_temperature'.")

            except requests.exceptions.RequestException:
                pass # No core temp stream or error fetching it

            # --- Fetch Lap Data for the race itself ---
            laps_url = f"https://intervals.icu/api/v1/activity/{activity['id']}/laps.csv"
            try:
                laps_response = requests.get(laps_url, auth=("API_KEY", api_key))
                laps_response.raise_for_status()
                laps_data = pd.read_csv(io.StringIO(laps_response.text))

                # Add lap data to race entry (e.g., for each lap, add its details)
                for index, lap in laps_data.iterrows():
                    lap_prefix = f"Lap {lap['lap']}"
                    race_raw_metrics[f'{lap_prefix} - Elapsed Time (s)'] = lap.get('elapsed_time')
                    race_raw_metrics[f'{lap_prefix} - Moving Time (s)'] = lap.get('moving_time')
                    race_raw_metrics[f'{lap_prefix} - Distance (m)'] = lap.get('distance')
                    race_raw_metrics[f'{lap_prefix} - Average Speed (m/s)'] = lap.get('average_speed')
                    race_raw_metrics[f'{lap_prefix} - Average Watts'] = lap.get('average_watts')

            except requests.exceptions.RequestException:
                pass # No lap data or error fetching it, continue without laps


            for days_to_race in prediction_horizons:
                prediction_date = race_date - timedelta(days=days_to_race)

                # Ensure prediction_date is not in the future relative to today
                if prediction_date > today:
                    continue

                # Create a new entry for each prediction point
                predictive_entry = {
                    'race_activity_id': activity['id'],
                    'race_date': race_date_str,
                    'prediction_date': prediction_date.isoformat(),
                    'days_to_race': days_to_race,
                    'target_performance_score': None # This will be calculated later after all races are processed
                }
                predictive_entry.update(race_raw_metrics) # Add all race-specific raw metrics

                # --- Fetch Pre-Prediction Wellness Data ---
                # 7-day window relative to prediction_date
                seven_days_before_prediction = prediction_date - timedelta(days=7)
                wellness_url_7_day = f"https://intervals.icu/api/v1/athlete/{athlete_id}/wellness?oldest={seven_days_before_prediction.isoformat()}&newest={prediction_date.isoformat()}"
                try:
                    wellness_response_7_day = requests.get(wellness_url_7_day, auth=("API_KEY", api_key))
                    wellness_response_7_day.raise_for_status()
                    wellness_data_7_day = wellness_response_7_day.json()

                    avg_resting_hr_7 = [e['restingHR'] for e in wellness_data_7_day if e.get('restingHR') is not None]
                    avg_hrv_7 = [e['hrv'] for e in wellness_data_7_day if e.get('hrv') is not None]
                    avg_sleep_secs_7 = [e['sleepSecs'] for e in wellness_data_7_day if e.get('sleepSecs') is not None]

                    predictive_entry['pred_Avg_Resting_HR_7_day'] = round(sum(avg_resting_hr_7) / len(avg_resting_hr_7), 2) if avg_resting_hr_7 else None
                    predictive_entry['pred_Avg_HRV_7_day'] = round(sum(avg_hrv_7) / len(avg_hrv_7), 2) if avg_hrv_7 else None
                    predictive_entry['pred_Avg_Sleep_hours_7_day'] = round(sum(avg_sleep_secs_7) / len(avg_sleep_secs_7) / 3600, 2) if avg_sleep_secs_7 else None

                except requests.exceptions.RequestException:
                    pass # No wellness data for this period or error fetching it

                # 14-day window relative to prediction_date
                fourteen_days_before_prediction = prediction_date - timedelta(days=14)
                wellness_url_14_day = f"https://intervals.icu/api/v1/athlete/{athlete_id}/wellness?oldest={fourteen_days_before_prediction.isoformat()}&newest={prediction_date.isoformat()}"
                try:
                    wellness_response_14_day = requests.get(wellness_url_14_day, auth=("API_KEY", api_key))
                    wellness_response_14_day.raise_for_status()
                    wellness_data_14_day = wellness_response_14_day.json()

                    avg_resting_hr_14 = [e['restingHR'] for e in wellness_data_14_day if e.get('restingHR') is not None]
                    avg_hrv_14 = [e['hrv'] for e in wellness_data_14_day if e.get('hrv') is not None]
                    avg_sleep_secs_14 = [e['sleepSecs'] for e in wellness_data_14_day if e.get('sleepSecs') is not None]

                    predictive_entry['pred_Avg_Resting_HR_14_day'] = round(sum(avg_resting_hr_14) / len(avg_resting_hr_14), 2) if avg_resting_hr_14 else None
                    predictive_entry['pred_Avg_HRV_14_day'] = round(sum(avg_hrv_14) / len(avg_hrv_14), 2) if avg_hrv_14 else None
                    predictive_entry['pred_Avg_Sleep_hours_14_day'] = round(sum(avg_sleep_secs_14) / len(avg_sleep_secs_14) / 3600, 2) if avg_sleep_secs_14 else None

                except requests.exceptions.RequestException:
                    pass # No wellness data for this period or error fetching it

                # --- Fetch PMC Data (day before prediction_date) ---
                day_before_prediction = prediction_date - timedelta(days=1)
                pmc_url = f"https://intervals.icu/api/v1/athlete/{athlete_id}/wellness?oldest={day_before_prediction.isoformat()}&newest={day_before_prediction.isoformat()}"
                try:
                    pmc_response = requests.get(pmc_url, auth=("API_KEY", api_key))
                    pmc_response.raise_for_status()
                    pmc_data = pmc_response.json()
                    if pmc_data:
                        predictive_entry['pred_Fitness_CTL'] = pmc_data[0].get('ctl')
                        predictive_entry['pred_Fatigue_ATL'] = pmc_data[0].get('atl')
                        predictive_entry['pred_Form_Ramp_Rate'] = pmc_data[0].get('rampRate')
                except requests.exceptions.RequestException:
                    pass # No PMC data for this day or error fetching it

                # --- Fetch Pre-Prediction Training Metrics (EF, Time Above 38C, Moving Time) ---
                # 1-week pre-prediction (no taper)
                time_above_38c_1w, avg_ef_1w, total_moving_time_1w = get_pre_race_training_metrics(athlete_id, api_key, prediction_date, 7, taper_days=0)
                predictive_entry['pred_Time_Above_38C_1w'] = time_above_38c_1w
                predictive_entry['pred_Total_Moving_Time_1w'] = round(total_moving_time_1w / 60, 2) if total_moving_time_1w is not None else None
                predictive_entry['pred_Avg_EF_1w'] = avg_ef_1w

                # 2-week pre-prediction (excluding taper)
                time_above_38c_2w, avg_ef_2w, total_moving_time_2w = get_pre_race_training_metrics(athlete_id, api_key, prediction_date, 14, taper_days=7)
                predictive_entry['pred_Time_Above_38C_2w'] = time_above_38c_2w
                predictive_entry['pred_Total_Moving_Time_2w'] = round(total_moving_time_2w / 60, 2) if total_moving_time_2w is not None else None
                predictive_entry['pred_Avg_EF_2w'] = avg_ef_2w

                # 3-week pre-prediction (excluding taper)
                _, avg_ef_3w, _ = get_pre_race_training_metrics(athlete_id, api_key, prediction_date, 21, taper_days=7)
                predictive_entry['pred_Avg_EF_3w'] = avg_ef_3w

                # 4-week pre-prediction (excluding taper)
                _, avg_ef_4w, _ = get_pre_race_training_metrics(athlete_id, api_key, prediction_date, 28, taper_days=7)
                predictive_entry['pred_Avg_EF_4w'] = avg_ef_4w

                # 5-week pre-prediction (excluding taper)
                _, avg_ef_5w, _ = get_pre_race_training_metrics(athlete_id, api_key, prediction_date, 35, taper_days=7)
                predictive_entry['pred_Avg_EF_5w'] = avg_ef_5w

                # 6-week pre-prediction (excluding taper)
                _, avg_ef_6w, _ = get_pre_race_training_metrics(athlete_id, api_key, prediction_date, 42, taper_days=7)
                predictive_entry['pred_Avg_EF_6w'] = avg_ef_6w

                predictive_dataset_rows.append(predictive_entry)

    # --- Data Analysis and Output ---
    if predictive_dataset_rows:
        df = pd.DataFrame(predictive_dataset_rows)

        # Calculate Performance Score for all races (on the entire dataset)
        # For normalization, we'll fill any missing values with 0, assuming that
        # missing data implies the worst possible score for that metric.

        # Normalize Speed - No NaNs expected here due to `activity.get('average_speed', 0)`
        speed_series = df['race_avg_speed_kmh']
        if speed_series.max() != speed_series.min():
            df['Normalized Speed'] = (speed_series - speed_series.min()) / (speed_series.max() - speed_series.min())
        else:
            df['Normalized Speed'] = 0

        # Normalize Efficiency Factor
        ef_series = df['race_efficiency_factor'].fillna(0)
        if ef_series.max() != ef_series.min():
            df['Normalized Efficiency Factor'] = (ef_series - ef_series.min()) / (ef_series.max() - ef_series.min())
        else:
            df['Normalized Efficiency Factor'] = 0

        # Normalize Power/Weight (W/kg)
        pw_series = df['race_power_weight_w_kg'].fillna(0)
        if pw_series.max() != pw_series.min():
            df['Normalized Power/Weight'] = (pw_series - pw_series.min()) / (pw_series.max() - pw_series.min())
        else:
            df['Normalized Power/Weight'] = 0

        # Simple performance score: sum of normalized metrics
        df['target_performance_score'] = df['Normalized Speed'] + df['Normalized Efficiency Factor'] + df['Normalized Power/Weight']

        # Drop intermediate normalization columns
        df = df.drop(columns=['Normalized Speed', 'Normalized Efficiency Factor', 'Normalized Power/Weight'])

        output_file = "predictive_dataset.csv"
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"Successfully saved predictive dataset to {output_file}")
        print("Sending ntfy notification for completion.")
        os.system(f"curl -d \"Predictive dataset generation complete. Saved to {output_file}\" https://ntfy.sh/myprefrontalcortex_GeminiCLI_channel")

    else:
        print("No races found to generate predictive dataset.")
        print("Sending ntfy notification for no races found.")
        os.system("curl -d \"Predictive dataset generation: No races found.\" https://ntfy.sh/myprefrontalcortex_GeminiCLI_channel")

except requests.exceptions.RequestException as e:
    print(f"An error occurred during data fetching: {e}")
except FileNotFoundError:
    print(f"Error: The file {output_file} was not found. Please ensure it has been generated.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
