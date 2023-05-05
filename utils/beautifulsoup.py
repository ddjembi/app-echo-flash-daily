import requests
from bs4 import BeautifulSoup
import http.client
import json
import re

def getPageContent(url):
    page_content = None
    error = None
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find the main content container (you may need to adjust the selector depending on the web page structure)
        main_content = soup.find('body')

        # Print the main content
        if main_content:
            # print(main_content.prettify())
            raw_text = main_content.get_text()

            # Remove extra spaces
            raw_text_no_space = text_without_extra_spaces = re.sub(r'\s{2,}', ' ', raw_text)

            # Remove consecutive newline characters, keeping a maximum of one
            page_content = re.sub(r'\n{2,}', '\n', raw_text_no_space)

        else:
            error = "The main content container could not be found."
    else:
        error = f"Failed to fetch the web page. Status code: {response.status_code}"
        
    return page_content, error