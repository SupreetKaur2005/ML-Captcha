from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time

# URL of the website
WEBSITE_URL = "http://127.0.0.1:5000"

# Email and Name to fill in the form
EMAIL = "test@example.com"
NAME = "Test User"

def main():
    # Set up the Selenium WebDriver
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    
    # Navigate to the website
    driver.get(WEBSITE_URL)
    
    try:
        # Locate and fill the email input field
        email_input = driver.find_element(By.NAME, "email")  # Replace 'email' with the actual 'name' or 'id' of the email input field
        email_input.send_keys(EMAIL)
        
        # Locate and fill the name input field
        name_input = driver.find_element(By.NAME, "name")  # Replace 'name' with the actual 'name' or 'id' of the name input field
        name_input.send_keys(NAME)
        
        # Locate and click the submit button
        submit_button = driver.find_element(By.XPATH, "//button[@type='submit']")  # Adjust the XPath if necessary
        submit_button.click()
        
        # Allow some time to observe the result before closing
        time.sleep(5)
    
    finally:
        # Close the browser
        driver.quit()

if __name__ == "__main__":
    main()