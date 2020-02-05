X = df.drop(['SUSPECT_ARRESTED_FLAG', 'SUSPECT_ARREST_OFFENSE', 'SEARCH_BASIS_INCIDENTAL_TO_ARREST_FLAG'], axis = 1)
y = df.SUSPECT_ARRESTED_FLAG

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 345)

enc = OneHotEncoder(handle_unknown='ignore')
scaler = StandardScaler()
X_train_num = X_train[['OBSERVED_DURATION_MINUTES', 'STOP_DURATION_MINUTES', 'SUSPECT_HEIGHT', 'SUSPECT_WEIGHT']]
X_train_cat = X_train.drop(['OBSERVED_DURATION_MINUTES', 'STOP_DURATION_MINUTES', 'SUSPECT_HEIGHT', 'SUSPECT_WEIGHT'], axis = 1)

X_train_enc = enc.fit_transform(X_train_cat).toarray()
names = enc.get_feature_names(X_train_cat.columns)

X_train_scale = scaler.fit_transform(X_train_num)

X_train_enc_df = pd.DataFrame(X_train_enc, columns = names)
X_train_scale_df = pd.DataFrame(X_train_scale, columns = X_train_num.columns)

X_train_new = pd.concat([X_train_enc_df, X_train_scale_df], axis = 1)

X_test_num = X_test[['OBSERVED_DURATION_MINUTES', 'STOP_DURATION_MINUTES', 'SUSPECT_HEIGHT', 'SUSPECT_WEIGHT']]
X_test_cat = X_test.drop(['OBSERVED_DURATION_MINUTES', 'STOP_DURATION_MINUTES', 'SUSPECT_HEIGHT', 'SUSPECT_WEIGHT'], axis = 1)

X_test_enc = enc.transform(X_test_cat).toarray()
X_test_scale = scaler.transform(X_test_num)

X_test_enc_df = pd.DataFrame(X_test_enc, columns = names)
X_test_scale_df = pd.DataFrame(X_test_scale, columns = X_test_num.columns)

X_test_new = pd.concat([X_test_enc_df, X_test_scale_df], axis = 1)

final_rf = RandomForestClassifier(criterion = 'gini', max_depth = 25, min_samples_leaf = 2, 
                                  min_samples_split = 4, n_estimators = 115)

final_rf.fit(X_train_new, y_train)
rf_train_pred = final_rf.predict(X_train_new)
rf_test_pred = final_rf.predict(X_test_new)

final_svc = SVC(C = 4.5, kernel = 'rbf', gamma = 'auto', max_iter = 10000)
final_svc.fit(X_train_new, y_train)
svc_train_pred = final_svc.predict(X_train_new)

final_xgb = XGBClassifier(learning_rate = 0.01, max_depth = 10, min_child_weight = 1, n_estimators = 50, subsample = 0.7)
final_xgb.fit(X_train_new, y_train)
xgb_train_pred = final_xgb.predict(X_train_new)
