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

    # Wait for startups to load (20s)
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
                        # Get outerHTML, using .text here will result in empty
                        key = tds[0].get_attribute("outerHTML") #key: Startups' attributes: Name, Founder, Hub, Size, etc.
                        value = tds[1].get_attribute("outerHTML") #value: the values of the mentioned keys
                        startup_info[key] = value
                        print(f"Scraped key: {key}, value: {value}") 
            except Exception as e:
                print(f"Error for {name}: {str(e)}")
            
            if startup_info not in startup_list:
                startup_list.append(startup_info) #avoid duplicates
            

        # Scroll down to the bottom of the page
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(3)  # Give time to load new content

        # Calculate new page height and compare with the last height for scrolling down
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height: 
            break  
        last_height = new_height  # Update last height to the new height

    return startup_list

#put in the website
driver = webdriver.Chrome()
url = 'https://www.de-hub.de/en/startupfinder/'  
driver.get(url)

# Let the page load fully
try:
    startups = scrape_startups()
except Exception as e:
    print(f"Error: {str(e)}")

driver.quit()

df = pd.DataFrame(startups) #make df of startups
print(df.head())

startups[0]

df_cleaned = df.replace(r'<\/?td>', '', regex=True) #using regex to remove HTML tags and clean data
df_cleaned.to_excel('startups_scrape_1902.xlsx', index=False)
df_cleaned