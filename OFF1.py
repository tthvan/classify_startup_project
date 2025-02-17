import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException

def scrape_startups():
    startup_list = []  

    last_height = driver.execute_script("return document.body.scrollHeight")  # Get initial page height

    while True:
        # Find all startup cards on the current page
        startup_all = driver.find_elements(By.CSS_SELECTOR, '.result-card.startup')

        for startup in startup_all:
            try:
                name = startup.find_element(By.CSS_SELECTOR, '.result-card-heading').text
            except NoSuchElementException:
                name = "N/A"  # Default value if the element is not found
            
            # Initialize a dictionary to store startup information with the name
            startup_info = {'name': name}

            try:
                # Find all rows (tr) in the tbody to extract the outerHTML of <td> pairs
                infos = startup.find_elements(By.CSS_SELECTOR, 'tbody tr')
                for info in infos:
                    tds = info.find_elements(By.CSS_SELECTOR, 'td')

                    # Only process rows that have two <td> elements
                    if len(tds) == 2:
                        key_html = tds[0].get_attribute("outerHTML")  # Get the full outer HTML of the first <td>
                        value_html = tds[1].get_attribute("outerHTML")  # Get the full outer HTML of the second <td>

                        # Add the key-value pair (both in outerHTML) to the startup_info dictionary
                        if key_html and value_html:
                            startup_info[key_html] = value_html

                # Check if this startup info is already in the list to avoid duplicates
                if startup_info not in startup_list:
                    startup_list.append(startup_info)
            except StaleElementReferenceException:
                print(f"Stale element encountered for startup: {name}, skipping.")

        # Scroll down to the bottom of the page
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)  # Wait for new startups to load

        # Calculate new page height and compare with the last height
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:  
            break
        last_height = new_height  # Update last height to the new height

    return startup_list

# Selenium setup
driver = webdriver.Chrome()
url = 'https://www.de-hub.de/en/startupfinder/'
driver.get(url)

# Wait for the results to load before starting to scrape
WebDriverWait(driver, 20).until(
    EC.presence_of_element_located((By.CLASS_NAME, 'result-list'))
)

# Call the scraping function to get data for multiple startups
all_startup_data = scrape_startups()

# Close the browser after scraping is done
driver.quit()

# Convert the list of dictionaries into a pandas DataFrame
if all_startup_data:
    df = pd.DataFrame(all_startup_data)

    # Print the DataFrame to check the structure
    print(df.head())

    # Optionally, export the DataFrame to an Excel file
    df.to_excel('all_startup_data_with_html.xlsx', index=False)

    # You can also export it to a CSV file if preferred
    df.to_csv('all_startup_data_with_html.csv', index=False)
else:
    print("No startup data found.")

all_startup_data 




####
# SINGLE START UP
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException, TimeoutException



def scrape_single_startup():
    startup_list = []  
    last_height = driver.execute_script("return document.body.scrollHeight")  # Get starting page height

    # Wait for startups to load
    try:
        WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.CSS_SELECTOR, '.result-card.startup')))
    except TimeoutException:
        print("Waiting for startup page to load")
        return startup_list
    
    # Scrape only the first startup
    startup_all = driver.find_elements(By.CSS_SELECTOR, '.result-card.startup')

    # Extract data for the first startup (just one)
    if startup_all:
        startup = startup_all[0]
        try:
            name = startup.find_element(By.CSS_SELECTOR, '.result-card-heading').text
            print(f"Scraping startup: {name}") 
        except NoSuchElementException:
            name = "N/A"  
            print("Could not find startup name.")
            
        startup_info = {'name': name}

        try:
            infos = startup.find_elements(By.CSS_SELECTOR, 'tbody tr')
            for info in infos:
                tds = info.find_elements(By.CSS_SELECTOR, 'td')
                if len(tds) == 2:
                    key = tds[0].text
                    value = tds[1].text
                    startup_info[key] = value
                    print(f"Scraped key: {key}, value: {value}")  # Debug print for each entry
        except Exception as e:
            print(f"Error while scraping details for {name}: {str(e)}")
        
        startup_list.append(startup_info)  # Add only the first startup's data

    return startup_list

driver = webdriver.Chrome()
url = 'https://www.de-hub.de/en/startupfinder/'  
driver.get(url)

# Let the page load fully and scrape just one startup
try:
    startup_data = scrape_single_startup()
except Exception as e:
    print(f"Error: {str(e)}")

driver.quit()

# If you want to store this data in a DataFrame and save it to Excel
df = pd.DataFrame(startup_data)
df_cleaned = df.replace(r'<\/?td>', '', regex=True)
df_cleaned.to_excel('single_startup_scrape.xlsx', index=False)
df_cleaned
