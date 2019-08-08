# Votando, Naturalmente

### **Description**: I obtained data of the manifestos of the largest parties running in Spain´s 2019 general election. After extracting such data, I use the Spacy library to clean, delete stopwards, lemmatize and tokenize the grouped text. I then run a vector function.

terrorist attacks that have taken place since 1970. I then enriched my data using an OECD API and web scrapping to obtain information on the GDP growth rate of the USA each year, with the objective of finding out whether more terrorist attacks in the USA took place in years of economic decline or high growth. 

#### There are several files in this project:

##### **loading.py**: I acquire and the database of terrorist attacks.
##### **wrangling.py**: I clean the data, renaming columns, filling missing values, correcting types and selecting the columns I want to focus on, dropping the rest.
##### **api.py**: I extract and organise the information from the api.
##### **web_scraping.py**: I extract a necessary value that was not accessible from the API (data for 1970), using web scraping.
##### **analysing.py**: I bin the yearly growth rates, I calculate and count the total sums of casualties and attacks per year and I finally merge all the data together into a dataframe.
##### **visualizing.py**: I create a key chart to show either the overall growth rate impact on attacks and casualties or the evolution of terrorist attacks and casualties over the years. 
##### **pdf.py**: I create a pdf with the data obtained.

#### **Conclusion**: The data shows that attacks and casualties are indeed higher in years of very low or negative GDP growth than they are in other years. Nevertheless, the data also shows that casualties and attacks are higher in years of high growth than in years of moderate growth. Also, the data is highly influenced by certain years such as 2001 with the 9/11 terrorist strike in the USA (lead to verylow/negative GDP growth year). Overall more analysis is required to obtain significant conclusion regarding this topic.
