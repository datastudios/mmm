from statsmodels.tsa.seasonal import seasonal_decompose
import string
import re
import pandas as pd
import numpy as np
import holidays

def preprocess_media_data(file_path):
    """
    Preprocesses media data for BI and MMM purposes.

    Args:
        file_path (str): The path to the media data file.

    Returns:
        tuple: A tuple containing two dataframes:
            - media_bi_df: The preprocessed media data for BI purposes.
            - media_mmm_df: The preprocessed media data for MMM purposes.
    """
    # Clean and transform media data for BI purposes
    media_df = pd.read_csv(file_path)
    media_df.week_of = pd.to_datetime(media_df.week_of)
    media_df.channel = media_df.channel.str.lower().str.replace(' ', '_')
    media_df.rename(columns={'site_group_b2c_mktg_all_channels':'platform'}, inplace=True)
    media_df.platform = media_df.platform.str.lower().str.replace(' ', '_')\
            .str.replace('-', '').str.replace('__', '_').str.replace("\'", '').str.replace(",", '')
    media_df.platform = np.where(media_df.platform.str.contains('every'), 'everyday_with_rachael_ray', media_df.platform)
    media_df.platform = np.where(media_df.platform.str.contains('people_weekly_'), 'people_weekly', media_df.platform)
    media_df.platform = np.where(media_df.platform.str.contains('walt_disney_internet_group_espn_internet_ventures'), 'walt_disney_internet', media_df.platform)    
    media_df.drop(columns=['c_id', 'month_of', 'tv_media_cost'], inplace=True)
    media_df.rename(columns={'b2c_media_cost':'media_cost', 'b2c_impressions':'impressions'}, inplace=True)
    media_bi_df = media_df.copy()

    # Pivot and aggregate media data for MMM purposes
    media_df = media_df[['week_of', 'channel', 'media_cost']]
    media_df = media_df.pivot_table(index=['week_of'], columns='channel', values='media_cost', aggfunc='sum').reset_index()
    media_df.columns = [col + '_spend' if col != 'week_of' else col for col in media_df.columns]
    media_df.columns.name = None
    media_df = media_df.fillna(0)
    media_mmm_df = media_df
    return media_bi_df, media_mmm_df

def preprocess_leads_policies_data(file_path, min_week, max_week):
    """
    Preprocesses the leads and policies data.

    Args:
        file_path (str): The file path of the leads and policies data.

    Returns:
        tuple: A tuple containing the preprocessed dataframes:
            - leads_bi_df (pd.DataFrame): Superset of leads data for BI.
            - sales_bi_df (pd.DataFrame): Superset of sales data for BI.
            - policies_df (pd.DataFrame): Subset of sales policy volume data for MMM.
            - premiums_df (pd.DataFrame): Subset of sales premiums data for MMM.
            - leads_df (pd.DataFrame): Aggregated leads data by week.
    """

    print('in here again')
    # Load leads and policy data
    leads_df = pd.read_csv(file_path)

    # Cast to correct data types
    leads_df.date_of_request = pd.to_datetime(leads_df.date_of_request)
    leads_df.contact_date = pd.to_datetime(leads_df.contact_date)
    leads_df.close_date = pd.to_datetime(leads_df.close_date)

    # Extract week of request, contact and close and set the end of the week to be Saturday to align with media data
    leads_df['week_of_request'] = pd.to_datetime(leads_df.date_of_request.dt.to_period('W-SAT').dt.end_time.dt.date)
    leads_df['week_of_contact'] = pd.to_datetime(leads_df.contact_date.dt.to_period('W-SAT').dt.end_time.dt.date)
    leads_df['week_of_close'] = pd.to_datetime(leads_df.close_date.dt.to_period('W-SAT').dt.end_time.dt.date)

    # Set week_of to be the week of request to align with media data
    leads_df['week_of'] = leads_df.week_of_request

    leads_df = leads_df[(leads_df.week_of >= min_week)&(leads_df.week_of <= max_week)]

    # Superset of leads data for BI
    leads_bi_df = leads_df[['date_of_request', 'week_of', 'week_of_request', 'week_of_contact', 'week_of_close', 'close_reason', 'b2c_policies', 'b2c_slms_ap', 'b2c_slms_leads']]

    # Populate sales data when close reason is 'closed with sale'
    sales_df = leads_bi_df[(leads_df.close_reason == 'closed with sale')]

    # Cases where close reason is 'closed with sale' but no sales data is available set with mean of annual premium
    mean_b2c_slms_ap = sales_df[sales_df.b2c_slms_ap != 0].b2c_slms_ap.mean()
    sales_df.loc[sales_df.b2c_slms_ap == 0, 'b2c_slms_ap'] = mean_b2c_slms_ap

    # Calculate days to close
    sales_df['days_to_close'] = (sales_df.week_of_close - sales_df.week_of_request).dt.days

    # Calculate premium monthly, annual and ltv
    sales_df['week_of'] = sales_df.week_of_request
    sales_df['premium_monthly'] = sales_df.b2c_slms_ap * 1.5
    sales_df['premium_annual'] = sales_df.premium_monthly * 12
    sales_df['premium_ltv'] = sales_df.premium_annual * 4

    # Superset of sales data for BI
    sales_bi_df = sales_df

    # Subset of sales policy volume data for MMM
    policies_df = sales_bi_df[['week_of', 'b2c_policies']]

    # Subset of sales premiums data for MMM
    premiums_df = sales_bi_df[['week_of', 'premium_monthly', 'premium_annual', 'premium_ltv']]

    # Aggregate data by week
    policies_df = policies_df.groupby(['week_of']).sum().reset_index().rename(columns={'b2c_policies':'policies'})
    premiums_df = premiums_df.groupby(['week_of']).sum().reset_index().rename(columns={'b2c_slms_ap':'premiums'})
    leads_df = leads_bi_df[['week_of', 'b2c_slms_leads']].groupby(['week_of']).sum().reset_index().rename(columns={'b2c_slms_leads':'leads'})

    return leads_bi_df, sales_bi_df, policies_df, premiums_df, leads_df


def seasonal_decomposition(df, idx, col, model_):
    """
    Perform seasonal decomposition on a given DataFrame.

    Parameters:
    - df: DataFrame
        The input DataFrame.
    - idx: str
        The column name to be used as the index.
    - col: str
        The column name to be decomposed.
    - model_: str, optional
        The type of seasonal decomposition model to use. Default is "additive".

    Returns:
    - decomposition: DecomposeResult
        The result of the seasonal decomposition.
    """
    df = df.set_index(idx)
    decomposition = seasonal_decompose(df[col], model=model_)
    return decomposition

def standardize_column_names(df):
    """
    Standardizes the column names of a DataFrame by converting them to lowercase,
    replacing punctuation with underscores, and removing leading/trailing underscores.

    Args:
        df (pd.DataFrame): The DataFrame whose column names need to be standardized.

    Returns:
        pd.DataFrame: The DataFrame with standardized column names.

    Example:
        >>> df = pd.DataFrame({'First Name': [1, 2], 'Last Name': [3, 4]})
        >>> df = standardize_column_names(df)
        >>> print(df.columns)
        Index(['first_name', 'last_name'], dtype='object')
    """
    new_cols = []
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))

    for c in df.columns.to_list():
        c_mod = c.lower()
        c_mod = c_mod.translate(translator)
        c_mod = '_'.join(c_mod.split(' '))
        if c_mod[-1] == '_':
            c_mod = c_mod[:-1]
        c_mod = re.sub(r'\_+', '_', c_mod)
        c_mod = re.sub('_+$', '', c_mod)
        new_cols.append(c_mod)
    df.columns = new_cols
    return df


def get_holidays(media_mmm_df):
    """
    Generate holiday dataframes for Business Intelligence (BI) and MMM (Media Mix Modeling).

    Parameters:
    media_mmm_df (DataFrame): The media data DataFrame.

    Returns:
    tuple: A tuple containing two DataFrames - holidays_bi_df and holidays_mmm_df.
        - holidays_bi_df: DataFrame containing holidays matching media dates for BI.
        - holidays_mmm_df: DataFrame containing holiday dummy variables for MMM.
    """
    # Get min and max weeks from media data
    min_week = media_mmm_df.week_of.min()
    max_week = media_mmm_df.week_of.max()

    # Get min and max years from media data
    min_year = min_week.year
    max_year = max_week.year + 1

    # Create years list to iterate over for holidays
    years_list = [year for year in range(min_year, max_year)]

    # Generate a dataframe of holidays matching media dates for BI
    holiday_list = []
    for holiday in holidays.USA(years=years_list).items():
        holiday_list.append(holiday)
    holidays_df = pd.DataFrame(holiday_list, columns=["date", "holiday"])
    holidays_df.date = pd.to_datetime(holidays_df.date)
    holidays_df['week_of_holiday'] = pd.to_datetime(holidays_df.date.dt.to_period('W-SAT').dt.end_time).dt.date
    holidays_df['week_of_holiday'] = pd.to_datetime(holidays_df['week_of_holiday'])
    holidays_df = holidays_df.sort_values(by='date')
    holidays_df = holidays_df[['week_of_holiday', 'holiday']].rename(columns={'week_of_holiday':'week_of'})
    holidays_df = holidays_df[(holidays_df.week_of >= min_week)&(holidays_df.week_of <= max_week)]
    holidays_bi_df = holidays_df.copy()
    holidays_bi_df

    # Generate holiday dummy variables for MMM
    holidays_mmm_df = pd.get_dummies(holidays_df, dtype=int)
    holidays_mmm_df = standardize_column_names(holidays_mmm_df)
    
    return holidays_bi_df, holidays_mmm_df
