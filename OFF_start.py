######################################
#           WEBSCRAPING         #
######################################
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time
import pandas as pd

def scrape_startups():
    startup_list = []  
    last_height = driver.execute_script("return document.body.scrollHeight")  # Get current page height

    # Wait for startups to load
    try:
        WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.CSS_SELECTOR, '.result-card.startup')))
    except TimeoutException:
        print("Cannot load in 20 secs")
        return startup_list
    
    #Find the large element body that includes information on all startups
    while True:
        startup_all = driver.find_elements(By.CSS_SELECTOR, '.result-card.startup')
    
    #loop through and scrape info of each startup in the body
        for startup in startup_all:
            try:
                name = startup.find_element(By.CSS_SELECTOR, '.result-card-heading').text
                print(f"Scraping startup: {name}") 
            except NoSuchElementException:
                name = "N/A"  
                print("Cannot find startup name")
                
            startup_info = {'name': name}

            try:
                infos = startup.find_elements(By.CSS_SELECTOR, 'tbody tr')
                for info in infos:
                    tds = info.find_elements(By.CSS_SELECTOR, 'td') #tag 'td' is where infos are located
                    
                    if len(tds) == 2:
                        # Get outerHTML
                        key = tds[0].get_attribute("outerHTML")
                        value = tds[1].get_attribute("outerHTML")
                        startup_info[key] = value
                        print(f"Scraped key: {key}, value: {value}") 
            except Exception as e:
                print(f"Error while scraping details for {name}: {str(e)}")
            
            if startup_info not in startup_list:
                startup_list.append(startup_info)
            

        # Scroll down to the bottom of the page
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(3)  # Give time to load new content

        # Calculate new page height and compare with the last height for scrolling down
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height: 
            break  
        last_height = new_height  # Update last height to the new height

    return startup_list

#fetch the website
driver = webdriver.Chrome()
url = 'https://www.de-hub.de/en/startupfinder/'  
driver.get(url)

# Let the page load fully
try:
    startups = scrape_startups()
except Exception as e:
    print(f"Error: {str(e)}")

driver.quit()

df = pd.DataFrame(startups)
print(df.head())

df_cleaned = df.replace(r'<\/?td>', '', regex=True)
df_cleaned.to_excel('startups_scrape.xlsx', index=False)
df_cleaned

######################################
#           EDA         #
######################################
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Import dataset
df_full = pd.read_excel(r"C:\Users\Admin\Documents\Studio Code\Startup\Germany Startups Dataset.xlsx")
#A review of the dataset 
df_full.shape #Number of rows & columns
df_full.isna().sum() #Number of missing values
df_full.columns #Columns

df = df_full[[#'No.', 'Company',
        'Industry', 'Technology', 'Looking_for',
       'Funding_phase', 'Hub_affiliation', #'Location',
        'Size', 'Market',
       'B2B_or_B2C', 'decision', #'Startup founders', 'Website', 'Email'
        ]]

######################################
#        GRAPHICAL VISUALIZATION      #
######################################
#Import dataset (after encoding into dummy variables)
df = pd.read_excel(r"C:\Users\Admin\Documents\Studio Code\data1_after_encode.xlsx")


#Find index of each column for later reference
col_name = "Industry"
index_no = df.columns.get_loc(col_name)
index_no

#Variable: INDUSTRY 
ind_df = pd.DataFrame(df.iloc[:, 2:19]) #locate where the columns of industry go
ind_df = ind_df.apply(pd.to_numeric, errors='coerce') #change to numeric
counts = ind_df.sum() #sum all the '1s', meaning 'Yes'
sorted_counts=counts.sort_values(ascending=True)
# Create a bar plot for the counts
highlight_columns = ['SaaS'] #highest count
highlight_color = '#FFEA00'
other_colors = ['#0000ff', '#0044ff', '#0066ff', '#3388ff', '#55aaff','#77ccff']
sorted_colors = [highlight_color if column in highlight_columns else np.random.choice(other_colors) for column in sorted_counts.index]
plt.barh(sorted_counts.index, sorted_counts.values, color=sorted_colors)
plt.ylabel('Industry')
plt.xlabel('Number of startups')
plt.xticks(rotation=90)
plt.title("Count of Startups' Industries")
plt.show()


#Variable: TECHNOLOGY 
tech_df=pd.DataFrame(df.iloc[:,19:28]) #locate where the columns of technology go
tech_df = tech_df.apply(pd.to_numeric, errors='coerce') #change to numeric
counts_tech=tech_df.sum() #sum all the '1s' in each technology column
sorted_counts=counts_tech.sort_values(ascending=True)
highlight_columns = ['Software Development'] #highest
highlight_color = '#FFEA00'
other_colors = ['#0000ff', '#0044ff', '#0066ff', '#3388ff', '#55aaff','#77ccff']
sorted_colors = [highlight_color if column in highlight_columns else np.random.choice(other_colors) for column in sorted_counts.index]
plt.barh(sorted_counts.index, sorted_counts.values, color=sorted_colors)
plt.title('Count of Core Technologies')
plt.ylabel('Core Technology')
plt.xlabel('Number of startups')
plt.xticks(rotation=90)
plt.show()


#Variable: LOOKING FOR
needs_df = pd.DataFrame(df.iloc[:,28:33])
needs_df = needs_df.apply(pd.to_numeric, errors='coerce') #change to numeric
counts_needs = needs_df.apply(pd.value_counts)
max_index = counts_needs.loc[1].idxmax()  # Find the column with the highest count of '1'
highlight_color = '#FFEA00'
other_color = '#77ccff'  # Color for other columns
colors = [highlight_color if col == max_index else other_color for col in needs_df.columns]
plt.bar(needs_df.columns, counts_needs.loc[1].values, color=colors)
plt.title("Count of Startup Needs")
plt.xlabel('Looking for')
plt.ylabel('Number of startups')
plt.show()


#Variable: HUB
hub_df=pd.DataFrame(df.iloc[:,34:48])
hub_df = hub_df.apply(pd.to_numeric, errors='coerce') #change to numeric
counts_hub=hub_df.sum()
highlight_columns = ['Hub Frankfurt/Darmstadt']
highlight_color = '#FFEA00'
sorted_counts = counts_hub.sort_values(ascending=True)
sorted_colors = [highlight_color if column in highlight_columns else np.random.choice(other_colors) for column in sorted_counts.index]
other_colors = ['#0000ff', '#0044ff', '#0066ff', '#3388ff', '#55aaff','#77ccff']
plt.barh(sorted_counts.index, sorted_counts.values, color=sorted_colors)
#plt.barh(hub_df.columns, counts_hub.values, color=[highlight_color if column in highlight_columns else np.random.choice(other_colors) for column in hub_df.columns])
plt.title("Hub Affliations of Startups")
plt.ylabel('Germany Digital Hub')
plt.xlabel('Number of startups')
plt.xticks(rotation=90)
plt.show()


#Variable: MARKET
mkt=df.groupby('Market').count()['Company']
highlight_columns = ['International']
highlight_color = '#FFEA00'
other_colors = ['#0000ff', '#0044ff', '#0066ff', '#3388ff', '#55aaff','#77ccff']
plt.bar(mkt.index, mkt.values, color=[highlight_color if column in highlight_columns else np.random.choice(other_colors) for column in mkt.index])
plt.title("Markets of Startups")
plt.ylabel('Number of startups')
plt.xlabel('Market')
plt.show()


#Variable: MARKET
type_=df.groupby('B2B or B2C').count()['Company']
highlight_columns = ['B2B']
highlight_color = '#FFEA00'
other_colors = ['#0000ff', '#0044ff', '#0066ff', '#3388ff', '#55aaff','#77ccff']
plt.bar(type_.index, type_.values, color=[highlight_color if column in highlight_columns else np.random.choice(other_colors) for column in type_.index])
plt.title("Types of Business (B2B/B2C)")
plt.xlabel('B2B or B2C')
plt.ylabel('Number of startups')
plt.show()


#Variable: FUNDING PHASE
funding=df.groupby('Funding phase').count().sort_values(by='Company',ascending=True)['Company']
highlight_columns = ['Seed']
highlight_color = '#FFEA00'
other_colors = ['#0000ff', '#0044ff', '#0066ff', '#3388ff', '#55aaff','#77ccff']
plt.barh(funding.index, funding.values, color=[highlight_color if column in highlight_columns else np.random.choice(other_colors) for column in funding.index])
plt.title("Funding phase of Startups")
plt.xlabel('Number of startups')
plt.ylabel('Funding phase')
plt.show()


#Variable: SIZE
size=df.groupby('Size').count().sort_values(by='Company',ascending=True)['Company']
highlight_columns = ['1-10']
highlight_color = '#FFEA00'
other_colors = ['#0000ff', '#0044ff', '#0066ff', '#3388ff', '#55aaff','#77ccff']
plt.barh(size.index, size.values, color=[highlight_color if column in highlight_columns else np.random.choice(other_colors) for column in size.index])
plt.title("Company size")
plt.xlabel('Number of startups')
plt.ylabel('Size')
plt.show()



#CORRELATION
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
pd.reset_option('display.max_rows')
pd.set_option('display.max_columns', 10)

#Get the useful columns for correlation
df = pd.read_excel(r"C:\Users\Admin\Documents\Studio Code\data2_after_imputed.xlsx")
df

cate_col = df[['AdTech',
'Cybersecurity',
'Digital Health',
'Digital Chemistry',
'E-Commerce',
'Education',
'FinTech',
'InsurTech',
'LegalTech',
'Logistics',
'MediaTech',
'Mobility',
'SaaS',
'Smart Infrastructure',
'Smart Systems',
'Cross-industry',
'Artificial intelligence',
'Blockchain',
'Robotics',
'Virtual Reality',
'Hardware',
'Software Development',
'Data Analytics',
'Internet of Things',
'Partner',
'Financing',
'Talents',
'Mentoring',
'funding_no',
'Hub Berlin',
'Hub Cologne',
'Hub Dresden/Leipzig',
'Hub Dortmund',
'Hub Frankfurt/Darmstadt',
'Hub Hamburg',
'Hub Karlsruhe',
'Hub Mannheim/Ludwigshafen',
'Hub Munich',
'Hub Potsdam',
'Hub Nuremberg/Erlangen',
'Hub Stuttgart',
'Not part of the network yet',
'size_no',
'market_no',
'b2b',
'b2c',
'num_headquart',
'dist_bin_no',
'num_of_founder',
'num_of_female',
'percent_female',
'high_potential'
]]

cate_col

################
# CRAMER'S V #
################

# Calculate the matrix of contingency coefficients
num_vars = cate_col.shape[1]

def cramers_v(var1, var2):
    # Step 1: Create the contingency table (crosstab) between the two variables
    crosstab = pd.crosstab(var1, var2, dropna=False)
    
    # Step 2: Run the chi-squared test
    chi2, _, _, _ = chi2_contingency(crosstab)
    
    # Step 3: Calculate the total observations
    n = crosstab.to_numpy().sum()
    
    # Step 4: Get the minimum dimension (rows or columns) and subtract 1
    min_dim = min(crosstab.shape) - 1
    
    # Step 5: Calculate Cramér’s V
    if min_dim == 0 or n == 0:
        return np.nan  # Avoid division by zero if the table is empty or has 1 category
    else:
        return np.sqrt(chi2 / (n * min_dim))

cramers_v(cate_col['percent_female'], cate_col['high_potential'])


rows = []
for col1 in cate_col.columns:
    col = []
    for col2 in cate_col.columns:
        cramers = cramers_v(cate_col[col1], cate_col[col2])
        col.append(round(cramers, 2))
    rows.append(col)

cramers_results = pd.DataFrame(rows, columns=cate_col.columns, index=cate_col.columns)
cramers_results

df1=pd.DataFrame(cramers_results,columns=cate_col.columns,index=cate_col.columns)

#Map Cramer's V heatmap
plt.figure(figsize=(20, 20))
sns.heatmap(df1, annot=False, fmt=".2f", linewidths=0.5)
plt.title("Cramér's V Heatmap", fontsize=18)
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.yticks(fontsize=10)
plt.savefig("cramers", bbox_inches='tight', dpi=100)


################
# Contingency matrix #
################

contingency_matrix = np.zeros((num_vars, num_vars))

for i, var1 in enumerate(cate_col.columns):
    for j, var2 in enumerate(cate_col.columns):
        if i != j:  # Exclude the same variables
            contingency_table = pd.crosstab(cate_col[var1], cate_col[var2])
            chi2, _, _, _ = chi2_contingency(contingency_table)
            contingency_coefficient = np.sqrt(chi2 / (chi2 + len(cate_col)))  # Calculate the contingency coefficient
            contingency_matrix[i, j] = contingency_coefficient

# Create a DataFrame to store the results
contingency_df = pd.DataFrame(contingency_matrix, columns=cate_col.columns, index=cate_col.columns)

plt.figure(figsize=(num_vars // 2, num_vars // 2))

# Create the heatmap using the contingency_df matrix
sns.heatmap(contingency_df, annot=True, fmt=".2f", linewidths=.5)

plt.title("Contingency Coefficient Heatmap")
plt.show()
plt.savefig("contingency_new.png", bbox_inches='tight', dpi=100)


######################################
#           IMPUTATION NULLS         #
######################################
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
pd.set_option('display.max_rows', None)

#Import dataset
df = pd.read_excel(r"C:\Users\Admin\Documents\Studio Code\Startup\Germany Startups Dataset.xlsx", sheet_name="Tách dummy")

#Check NA values
df.isnull().sum()

#Use iterative impute to impute missing values
imputer = IterativeImputer(random_state=100)
df_train=df.loc[:,['AdTech', 'Cybersecurity', #Pick the columns to impute
       'Digital Health', 'Digital Chemistry', 'E-Commerce', 'Education',
       'FinTech', 'InsurTech', 'LegalTech', 'Logistics', 'MediaTech',
       'Mobility', 'SaaS', 'Smart Infrastructure', 'Smart Systems',
       'Cross-industry', 'Artificial intelligence', 'Blockchain',
       'Robotics', 'Virtual Reality', 'Hardware', 'Software Development',
       'Data Analytics', 'Internet of Things', 'Partner',
       'Financing', 'Talents', 'Mentoring',
       'Hub Berlin', 'Hub Cologne', 'Hub Dresden/Leipzig', 'Hub Dortmund',
       'Hub Frankfurt/Darmstadt', 'Hub Hamburg', 'Hub Karlsruhe',
       'Hub Mannheim/Ludwigshafen', 'Hub Munich', 'Hub Potsdam',
       'Hub Nuremberg/Erlangen', 'Hub Stuttgart', 
       'Not part of the network yet','num_of_hub','b2b', 'b2c','size_no', 'funding_no', 'market_no',
       'num_headquart', 'dist',
        'dist_bin_no', 'num_of_founder', 'num_of_female', 'percent_female', 'high_potential'
    ]]

imputer.fit(df_train) #Fit the model for iterative imputing
df_imputed = imputer.transform(df_train)
df_imputed = pd.DataFrame(df_imputed, columns = df_train.columns)
df_imputed = df_imputed.round()
df_imputed.isnull().sum()

df_imputed.to_excel('data2_after_imputed.xlsx', index=False)


######################################
#           FEATURE SELECTION         #
######################################
from sklearn.feature_selection import chi2
X = df_imputed
y = df['decision_no']

#Use chi-squared to examine the association of 
chi_scores, p_values = chi2(X, y)
chi2_df = pd.DataFrame({'Feature': X.columns, 'Chi2': chi_scores, 'P-value': p_values})

chi_values = pd.Series(chi_scores, index=X.columns)
chi_values.sort_values(ascending=True, inplace=True)

plt.figure(figsize=(10, 10))
chi_values.plot.barh()
plt.xlabel('Chi-squared Value')
plt.ylabel('Features')
plt.title('Chi-squared Test Results')
plt.show()

df_feature_selection = df_imputed[[#'AdTech', 
        'Cybersecurity', #'Digital Health', 
        'Digital Chemistry',
       #'E-Commerce', 
       'Education', 'FinTech', 'InsurTech', #'LegalTech',
       'Logistics', 'MediaTech', 'Mobility', #'SaaS', 
       'Smart Infrastructure',
       'Smart Systems', #'Cross-industry', 
       'Artificial intelligence',
       #'Blockchain', 
       'Robotics', #'Virtual Reality', 
       'Hardware',     
       'Software Development', 'Data Analytics', 'Internet of Things',
       'Partner', 'Financing', 'Talents', 'Mentoring', 'Hub Berlin',
       'Hub Cologne', #'Hub Dresden/Leipzig', 
       'Hub Dortmund',        
       'Hub Frankfurt/Darmstadt', #'Hub Hamburg', 'Hub Karlsruhe', 'Hub Mannheim/Ludwigshafen', 
       'Hub Munich', 'Hub Potsdam',    
       #'Hub Nuremberg/Erlangen', 
       'Hub Stuttgart',
       'Not part of the network yet', #'B2B', 
       'B2C', 'size_no', 'funding_no',
       #'mkt_no'
       ]]

df_feature_selection['decision_no'] = df['decision_no']

######################################
#           MODELLING         #
######################################
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, make_scorer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot
from sklearn.metrics import precision_recall_curve

#Split the dataset into training and testing
df_train, df_test = train_test_split(df_feature_selection, train_size = 0.7, test_size = 0.3, random_state = 100)
y_train = df_train['decision_no']
X_train = df_train.drop(columns='decision_no')
y_test=df_test['decision_no']
X_test = df_test.drop(columns='decision_no')

##################
#LOGISTIC REGRESSION
##################
model_lr = LogisticRegression(random_state=42)
model_lr.fit(X_train, y_train)
predicted_probs = model_lr.predict_proba(X_test)[:, 1]  # Predicted probabilities for class 0
y_pred = model_lr.predict(X_test)
lr_score = accuracy_score(y_test, y_pred)

#Evaluation results
report = classification_report(y_test,y_pred, target_names=['Class 0', 'Class 1'])
class1_metrics = report.split('\n\n')[1]
class1_metrics

precision, recall, f1_score, support = precision_recall_fscore_support(y_test, y_pred, pos_label=1, average='binary')
class1_metrics = pd.DataFrame({
    'Metric': ['Precision', 'Recall', 'F1-Score', 'Support', 'Accuracy'],
    'Logistic Regression': [precision, recall, f1_score, support, lr_score]
})

##################
#DECISION TREE
##################
model_dt = DecisionTreeClassifier(criterion='entropy',random_state=42)
model_dt.fit(X_train,y_train)
dt_pred = model_dt.predict(X_test)
dt_score=accuracy_score(y_test, dt_pred)

#Evaluation results
precision_dt, recall, f1_score, support = precision_recall_fscore_support(y_test, dt_pred, pos_label=1, average='binary')
class1_metrics['Decision tree'] = [precision_dt, recall, f1_score, support, dt_score]
class1_metrics

#Feature importance
dt_fi = model_dt.feature_importances_
dt_fi_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': dt_fi})
dt_fi_df  = dt_fi_df.sort_values(by='Importance', ascending=True) #sort the variables regarding their importance to predictions

# Plotting Feature importance
plt.figure(figsize=(10, 6))
plt.barh(dt_fi_df['Feature'], dt_fi_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.show()


##################
#RANDOM FOREST
##################
model_rf =RandomForestClassifier(random_state=42)
model_rf.fit(X_train,y_train)
rf_pred = model_rf.predict(X_test)
rf_score=accuracy_score(y_test, rf_pred)

#Evaluation results
precision_rf, recall, f1_score, support = precision_recall_fscore_support(y_test, rf_pred, pos_label=1, average='binary')
class1_metrics['Random forest'] = [precision_rf, recall, f1_score, support, rf_score]
class1_metrics

#Feature importance
fi_rf = model_rf.feature_importances_
fi_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': fi_rf})
fi_df  = fi_df.sort_values(by='Importance', ascending=True) #Sort from high importance to low

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.barh(fi_df ['Feature'], fi_df ['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.show()


##################
#SVM
##################
model_svm = SVC(kernel='rbf', probability=True, random_state=42)
model_svm.fit(X_train, y_train)
svm_pred = model_svm.predict(X_test)
svm_score=accuracy_score(y_test, svm_pred)

#Evaluation results
precision_svm, recall, f1_score, support = precision_recall_fscore_support(y_test, svm_pred, pos_label=1, average='binary')
class1_metrics['SVM'] = [precision_svm, recall, f1_score, support, svm_score]

##################
#Naive Bayes
##################
model_nb = BernoulliNB()
model_nb.fit(X_train, y_train)
nb_pred = model_nb.predict(X_test)
nb_score=accuracy_score(y_test, nb_pred)
nb_score

#Evaluation results
precision_nb, recall, f1_score, support = precision_recall_fscore_support(y_test, nb_pred, pos_label=1, average='binary')
class1_metrics['Naive bayes'] = [precision_nb, recall, f1_score, support, nb_score]
class1_metrics = class1_metrics[class1_metrics['Metric'] != 'Support']
class1_metrics


######################################
#           ROC-AUC EVALUATION         #
######################################
#Calculate ROC-AUC for each model
#for RF
rf_prob = model_rf.predict_proba(X_test)
rf_probs = rf_prob[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)

#for SVM
svm_prob = model_svm.predict_proba(X_test)
svm_probs = svm_prob[:, 1]
fpr_svm, tpr_svm, _ = roc_curve(y_test, svm_probs)

#for logistic
log_prob = model_lr.predict_proba(X_test)
log_probs = log_prob[:, 1]
fpr, tpr, _ = roc_curve(y_test, log_probs)

#for DT
dt_prob = model_dt.predict_proba(X_test)
dt_probs = dt_prob[:, 1]
fpr_dt, tpr_dt, _ = roc_curve(y_test, dt_probs)

#for NB
nb_prob = model_nb.predict_proba(X_test)
nb_probs = nb_prob[:, 1]
fpr_nb, tpr_nb, _ = roc_curve(y_test, nb_probs)

#Add ROC-AUC to each model in result dataframe
log_auc=roc_auc_score(y_test, log_probs)
dt_auc=roc_auc_score(y_test, dt_probs)
rf_auc=roc_auc_score(y_test, rf_probs)
svm_auc=roc_auc_score(y_test, svm_probs)
nb_auc=roc_auc_score(y_test, nb_probs)

auc_row={
    'Metric': 'AUC',
    'Logistic Regression': log_auc,
    'Decision tree': dt_auc,
    'Random forest': rf_auc,
    'SVM': svm_auc,
    'Naive bayes': nb_auc
}

auc_row_df = pd.DataFrame([auc_row])
class1_metrics = pd.concat([class1_metrics, auc_row_df], ignore_index=True)
class1_metrics


#############
#PLOT AUC-ROC
#############
pyplot.plot([0, 1], [0, 1], linestyle='--')
pyplot.plot(fpr_rf, tpr_rf, color='green', marker='.', label=f'Random forest (auc = {rf_auc:.3f})')
pyplot.plot(fpr_svm, tpr_svm, color='red', marker='.', label=f'SVM (auc = {svm_auc:.3f})')
pyplot.plot(fpr, tpr, marker='.', label=f'Logistic regression (auc = {log_auc:.3f})')
pyplot.plot(fpr_nb, tpr_nb, color='pink', marker='.', label=f'Naive Bayes (auc = {nb_auc:.3f})')
pyplot.plot(fpr_dt, tpr_dt, color='navy', marker='.', label=f'Decision tree (auc = {dt_auc:.3f})')

pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.legend()
pyplot.show()


#############
#PLOT PR-AUC
#############
# Calculate precision-recall values for each model
precision_log, recall_log, _ = precision_recall_curve(y_test, log_probs)
precision_dt, recall_dt, _ = precision_recall_curve(y_test, dt_probs)
precision_rf, recall_rf, _ = precision_recall_curve(y_test, rf_probs)
precision_svm, recall_svm, _ = precision_recall_curve(y_test, svm_probs)
precision_nb, recall_nb, _ = precision_recall_curve(y_test, nb_probs)

# Calculate AUC for precision-recall curves
pr_auc_log = auc(recall_log, precision_log)
pr_auc_dt = auc(recall_dt, precision_dt)
pr_auc_rf = auc(recall_rf, precision_rf)
pr_auc_svm = auc(recall_svm, precision_svm)
pr_auc_nb = auc(recall_nb, precision_nb)

# Plot Precision-Recall curves
plt.plot(recall_rf, precision_rf, color='green', marker='.', label=f'Random forest (auc = {pr_auc_rf:.3f})')
plt.plot(recall_svm, precision_svm, color='red', marker='.', label=f'SVM (auc = {pr_auc_svm:.3f})')
plt.plot(recall_log, precision_log, marker='.', label=f'Logistic regression (auc = {pr_auc_log:.3f})')
plt.plot(recall_nb, precision_nb, color='pink', marker='.', label=f'Naive Bayes (auc = {pr_auc_nb:.3f})')
plt.plot(recall_dt, precision_dt, color='navy', marker='.', label=f'Decision tree (auc = {pr_auc_dt:.3f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.show()


######################################
#           VALIDATION         #
######################################
from sklearn.model_selection import cross_val_score, KFold

#kfold
X = df_feature_selection.drop(columns=['decision_no'], axis=1)
y = df_feature_selection['decision_no']

k = 10
scores_lr = cross_val_score(model_lr, X, y, cv=k, scoring='roc_auc')
scores_dt = cross_val_score(model_dt, X, y, cv=k, scoring='roc_auc')
scores_rf = cross_val_score(model_rf, X, y, cv=k, scoring='roc_auc')
scores_svm = cross_val_score(model_svm, X, y, cv=k, scoring='roc_auc')
scores_nb = cross_val_score(model_nb, X, y, cv=k, scoring='roc_auc')

df_validate = pd.DataFrame({'Fold': range(1, k+1), 'LR': scores_lr, 'DT': scores_dt,'RF': scores_rf, 'SVM': scores_svm, 'NB': scores_nb}) #Make a df storing all
df_validate

#Add the mean and standard dev. of each validating results to df
mean_row = df_validate[['LR', 'DT', 'RF', 'SVM', 'NB']].mean()
std_row = df_validate[['LR', 'DT', 'RF', 'SVM', 'NB']].std()
df_validate.loc['Mean'] = [None] + list(mean_row)  # None for the 'Fold' column
df_validate.loc['Std Dev'] = [None] + list(std_row)
df_validate


###############
#STRATIFIED K-FOLD
###############
from sklearn.model_selection import StratifiedKFold

n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
roc_auc_scores = []

for fold_num, (train_index, test_index) in enumerate(skf.split(X_scaled, y), start=1):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]  # Use X_scaled instead of X
    y_train, y_test = y[train_index], y[test_index]
    model_rf.fit(X_train, y_train)
    y_probs = model_rf.predict_proba(X_test)[:, 1] #Predict probability for class 1
    roc_auc = roc_auc_score(y_test, y_probs)

    roc_auc_scores.append(roc_auc)

# Calculate mean and standard deviation of ROC-AUC scores
mean_roc_auc = np.mean(roc_auc_scores)
std_deviation = np.std(roc_auc_scores)

roc_auc_df2 = pd.DataFrame({
    'Fold': list(range(1, n_splits + 1)),
    'ROC-AUC': roc_auc_scores
})
mean_std_df = pd.DataFrame({
    'Fold': ['Mean', 'Std Dev'],
    'ROC-AUC': [mean_roc_auc, std_deviation]
})
roc_auc_df2 = pd.concat([roc_auc_df2, mean_std_df], ignore_index=True)

roc_auc_df2




