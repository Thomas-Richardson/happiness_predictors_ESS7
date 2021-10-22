import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data_raw = pd.read_csv(
    'https://raw.githubusercontent.com/Thomas-Richardson/Blog_post_data/main/ESS7.csv')

data = data_raw.drop(columns=['ess7_id', 'nuts1', 'nuts2', 'nuts3', 'ess7_reg'])

data.columns

data = data.rename(columns={'cntry': 'country', 'tvpol': 'daily_tv', 'ppltrst':
                            'people_trustworthy', 'pplfair': 'people_fair', 'pplhlp':
                            'people_helpful', 'lrscale': 'political_orientation_lr', 'stfeco':
                            'satisfaction_economy', 'stfgov': 'satisfaction_government', 'stfdem':
                            'satisfaction_democracy_here', 'stfedu':
                            'satisfaction_education_system', 'stfhlth': 'state_of_healthcare',
                            'sclmeet': 'meet_friendsfam_often', 'inprdsc': 'people_to_confide_in',
                            'sclact': 'social_life', 'crmvct': 'burglary_assault_victim_5y',
                            'aesfdrk': 'fear_area_afterdark', 'hlthhmp': 'disability', 'rlgblg':
                            'religious', 'dscrgrp': 'oppressed_group', 'ctzcntr': 'citizen',
                            'blgetmg': 'minority', 'etfruit': 'eat_fruit', 'eatveg': 'eat_veg',
                            'dosprt': 'sport', 'cgtsmke': 'smoker', 'alcfreq': 'alcohol_often',
                            'alcbnge': 'binge_drinking', 'slprl': 'sleep_restless_past_week',
                            'fltlnl': 'lonely_past_week', 'cnfpplh': 'family_conflict_childhood',
                            'fnsdfml': 'childhood_financial_problems', 'gndr': 'sex', 'agea':
                            'age', 'hincfel': 'feeling_about_income', 'atncrse':
                            'improve_knowlege', 'maritalb': 'marital_status', 'dvrcdeva':
                            'divorced', 'chldhm': 'kids_at_home', 'domicil': 'area_type',
                            'eduyrs': 'years_education', 'wkhct': 'hours_overtime_excl', 'wkhtot':
                            'hours_overtime_incl', 'nacer2': 'industry', 'uemp3m':
                            'ever_unemployed', 'hinctnta': 'income_decile', 'psppsgv':
                            'have_say_politics', 'psppipl': 'have_influence_politics', 'cptppol':
                            'confident_participate_politics', 'ptcpplt': 'politicians_listen',
                            'trstprl': 'trust_parliament', 'trstlgl': 'trust_legal_system',
                            'trstplc': 'trust_police', 'trstplt': 'trust_politicans', 'trstprt':
                            'trust_political_parties'})
# data.columns

data.political_orientation_lr \
    .value_counts(normalize=True, dropna=False) \
    .round(2) \
    .to_frame() \
    .reset_index() \
    .sort_values('index')  # % of a column that is taken up by each category

five_cols = ['social_life', 'citizen', 'binge_drinking', 'sleep_restless_past_week',
             'lonely_past_week', 'family_conflict_childhood', 'childhood_financial_problems',
             'sex', 'divorced', 'kids_at_home', 'area_type', 'ever_unemployed',
             'feeling_about_income', 'improve_knowlege']
data[five_cols] = data[five_cols].where(data[five_cols] < 5, np.nan)

seven_cols = ['marital_status', 'meet_friendsfam_often', 'daily_tv', 'people_to_confide_in',
              'burglary_assault_victim_5y', 'fear_area_afterdark', 'health', 'disability',
              'religious', 'oppressed_group', 'citizen', 'minority', 'smoker']
data[seven_cols] = data[seven_cols].where(data[seven_cols] < 7, np.nan)

ten_cols = ['happy', 'people_trustworthy', 'people_fair', 'people_helpful',
            'political_orientation_lr', 'satisfaction_education_system', 'state_of_healthcare',
            'satisfaction_economy', 'satisfaction_government', 'satisfaction_democracy_here',
            'eat_fruit', 'eat_veg', 'sport', 'alcohol_often', 'income_decile',
            'have_say_politics', 'have_influence_politics', 'confident_participate_politics',
            'politicians_listen', 'trust_parliament', 'trust_legal_system', 'trust_police',
            'trust_politicans', 'trust_political_parties']
data[ten_cols] = data[ten_cols].where(data[ten_cols] < 11, np.nan)

data = data.drop(columns='Unnamed: 63')
data.select_dtypes(include=np.number).columns.tolist()

data = data.dropna(
    subset=['happy'])  # obvious drop those for whom we have no target variable

data.to_csv('ESS7_cleaned1.csv', index=False)
