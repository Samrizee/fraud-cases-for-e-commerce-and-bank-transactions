import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta
from src.FeatureEnginerring import get_country, engineer_fraud_features
from src.Cleaning import clean_fraud_data 


mock_ip_ranges_df = pd.DataFrame({
    'lower_bound_ip_address': [0, 1000, 2000, 3000],
    'upper_bound_ip_address': [999, 1999, 2999, 3999],
    'country': ['CountryA', 'CountryB', 'CountryC', 'CountryD']
}).sort_values(by='lower_bound_ip_address').reset_index(drop=True)

class TestDataProcessing:

    def test_get_country_valid_ip(self):
        assert get_country(500, mock_ip_ranges_df) == 'CountryA'
        assert get_country(1500, mock_ip_ranges_df) == 'CountryB'
        assert get_country(2500, mock_ip_ranges_df) == 'CountryC'

    def test_get_country_boundary_ip(self):
        assert get_country(0, mock_ip_ranges_df) == 'CountryA'
        assert get_country(999, mock_ip_ranges_df) == 'CountryA'
        assert get_country(1000, mock_ip_ranges_df) == 'CountryB'
        assert get_country(1999, mock_ip_ranges_df) == 'CountryB'

    def test_get_country_unknown_ip(self):
        assert get_country(4000, mock_ip_ranges_df) == 'Unknown'
        assert get_country(-1, mock_ip_ranges_df) == 'Unknown'
        assert get_country(10000, mock_ip_ranges_df) == 'Unknown'

    def test_remove_duplicates(self):
        data = {'col1': [1, 2, 1, 3], 'col2': ['a', 'b', 'a', 'c']}
        df = pd.DataFrame(data)
        df_cleaned = df.drop_duplicates()
        assert len(df_cleaned) == 3
        assert df_cleaned.iloc[0]['col1'] == 1
        assert df_cleaned.iloc[1]['col1'] == 2
        assert df_cleaned.iloc[2]['col1'] == 3

    def test_datetime_conversion(self):
        data = {'signup_time': ['2023-01-01 10:00:00'], 'purchase_time': ['2023-01-01 11:00:00']}
        df = pd.DataFrame(data)
        
        df['signup_time'] = pd.to_datetime(df['signup_time'])
        df['purchase_time'] = pd.to_datetime(df['purchase_time'])
        assert isinstance(df['signup_time'].iloc[0], datetime)
        assert isinstance(df['purchase_time'].iloc[0], datetime)

    def test_ip_address_coercion_and_dropna(self):
        data = {
            'user_id': [1, 2, 3],
            'signup_time': ['2023-01-01 10:00:00', '2023-01-01 10:00:00', '2023-01-01 10:00:00'],
            'purchase_time': ['2023-01-01 11:00:00', '2023-01-01 11:00:00', '2023-01-01 11:00:00'],
            'purchase_value': [100, 200, 300],
            'device_id': ['d1', 'd2', 'd3'],
            'source': ['s1', 's2', 's3'],
            'browser': ['b1', 'b2', 'b3'],
            'sex': ['M', 'F', 'M'],
            'age': [25, 30, 35],
            'ip_address': [12345, 'invalid_ip', 67890],
            'class': [0, 1, 0]
        }
        df = pd.DataFrame(data)
        initial_len = len(df)
        df_cleaned = clean_fraud_data(df.copy()) 
        assert len(df_cleaned) == initial_len - 1 
        assert df_cleaned['ip_address'].dtype == np.int64 

    def test_engineered_fraud_features(self):
        data = {
            'user_id': [1, 1, 2, 2, 1],
            'device_id': ['A', 'A', 'B', 'B', 'C'],
            'purchase_time': [
                datetime(2023, 1, 1, 10, 0, 0),
                datetime(2023, 1, 1, 10, 30, 0),
                datetime(2023, 1, 1, 11, 0, 0),
                datetime(2023, 1, 1, 11, 15, 0),
                datetime(2023, 1, 1, 10, 45, 0) 
            ],
            'signup_time': [datetime(2023, 1, 1, 9, 0, 0)] * 5 
        }
        df = pd.DataFrame(data)
        df_processed = engineer_fraud_features(df.copy())

        expected_time_diff_user = pd.Series([0.0, 1800.0, 0.0, 900.0, 900.0], name='time_diff_user')
        pd.testing.assert_series_equal(df_processed['time_diff_user'], expected_time_diff_user, check_dtype=False)

        
        expected_time_diff_device_values = pd.Series([0.0, 1800.0, 0.0, 900.0, 0.0], name='time_diff_device')
        pd.testing.assert_series_equal(df_processed['time_diff_device'], expected_time_diff_device_values, check_dtype=False)

        
        expected_device_counts = pd.Series([2, 2, 2, 2, 1], name='device_transaction_count')
        pd.testing.assert_series_equal(df_processed['device_transaction_count'], expected_device_counts, check_dtype=False)

        
        expected_user_counts = pd.Series([3, 3, 2, 2, 3], name='user_transaction_count')
        pd.testing.assert_series_equal(df_processed['user_transaction_count'], expected_user_counts, check_dtype=False)

        expected_time_since_signup = pd.Series([3600.0, 5400.0, 7200.0, 8100.0, 6300.0], name='time_since_signup_seconds')
        pd.testing.assert_series_equal(df_processed['time_since_signup_seconds'], expected_time_since_signup, check_dtype=False)
