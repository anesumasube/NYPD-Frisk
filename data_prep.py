unclean_df = unclean_df.drop(['STOP_FRISK_ID', 'STOP_FRISK_DATE', 'Stop Frisk Time', 'YEAR2', 'ISSUING_OFFICER_COMMAND_CODE', 
                 'SUPERVISING_OFFICER_COMMAND_CODE', 'LOCATION_IN_OUT_CODE', 'RECORD_STATUS_CODE', 'JURISDICTION_CODE', 
                 'OFFICER_NOT_EXPLAINED_STOP_DESCRIPTION', 'SUMMONS_OFFENSE_DESCRIPTION', 'DEMEANOR_OF_PERSON_STOPPED', 
                 'SUSPECT_OTHER_DESCRIPTION','STOP_LOCATION_APARTMENT', 'STOP_LOCATION_ZIP_CODE', 'STOP_LOCATION_FULL_ADDRESS',
                'STOP_LOCATION_PREMISES_NAME', 'STOP_LOCATION_STREET_NAME', 'STOP_LOCATION_X', 'STOP_LOCATION_Y'], axis = 1)

unclean_df.SUSPECT_REPORTED_AGE.replace('(null)', '-15', inplace = True)
unclean_df.SUSPECT_REPORTED_AGE = unclean_df.SUSPECT_REPORTED_AGE.astype(float)
cut_labels = ['unknown', '0_17', '18_30', 'over_30']
cut_bins = [-16, -1, 17, 30, 100]
unclean_df['age_bin'] = pd.cut(unclean_df.SUSPECT_REPORTED_AGE, bins = cut_bins, labels = cut_labels)

unclean_df = unclean_df.drop('SUSPECT_REPORTED_AGE', axis = 1)

unclean_df = unclean_df[unclean_df['SUSPECT_HEIGHT'] != '(null)']
unclean_df.SUSPECT_HEIGHT = unclean_df.SUSPECT_HEIGHT.astype(float)

unclean_df = unclean_df[unclean_df['SUSPECT_WEIGHT'] != '(null)']
unclean_df.SUSPECT_WEIGHT = unclean_df.SUSPECT_WEIGHT.astype(float)

X = unclean_df.drop('SUSPECT_ARRESTED_FLAG', axis = 1)
y = unclean_df.SUSPECT_ARRESTED_FLAG