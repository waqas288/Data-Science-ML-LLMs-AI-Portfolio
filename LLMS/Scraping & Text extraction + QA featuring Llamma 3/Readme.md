# **Clinical Trial Data Extractor**  

## **Overview**  
The **Clinical Trial Data Extractor** is a Python-based tool designed to scrape and analyze clinical trial data from **PubMed**. It automates the retrieval of randomized controlled trials, extracts key trial details, and processes the information using an **LLM (Large Language Model)** for structured summarization.  

## **Features**  
- **PubMed Scraping:** Retrieves clinical trial data based on a keyword search.  
- **Automated Page Crawling:** Extracts all available trials using pagination.  
- **Trial Information Extraction:** Captures key details like title, link, and abstract.  
- **LLM Processing:** Summarizes trial data into structured formats.  
- **Logging & Error Handling:** Robust logging to track errors and process status.  

## **Installation**  

### **Prerequisites**  
Ensure you have Python **3.8+** installed on your system.  

### **Dependencies**  
Install the required packages using:  
```bash
pip install -r requirements.txt
```  

### **Environment Setup**  
Create a `.env` file in the project directory and add your **OpenRouter API key**:  
```bash
OPENROUTER_API_KEY=your_api_key_here
```  

## **Usage**  

### **Running the Scraper**  
Execute the script with:  
```bash
python clinical_trial_extractor.py
```  

### **Modifying Search Queries**  
Modify the `keyword` variable in the script to change the search query:  
```python
keyword = "Lung Cancer"
```  

## **Project Structure**  
```
│── clinical_trial_extractor.py  # Main script  
│── requirements.txt             # Dependencies  
│── .env                         # API keys (not included in the repo)  
│── clinical_trials_processing.log  # Log file  
```  

## **Functionality**  

### **1. Scraping PubMed**  
- **`search_pubmed(keyword, page)`** – Fetches search results for a given keyword.  
- **`get_total_pages(keyword)`** – Determines the total number of result pages.  
- **`extract_trial_info(html_content)`** – Extracts trial titles and links.  

### **2. Retrieving Full Article Text**  
- **`get_full_article_text(url)`** – Fetches and extracts the abstract.  

### **3. Processing with LLM**  
- **`process_trial_with_llm(trial_text)`** – Sends trial text to OpenRouter AI for structured extraction.  

### **4. Data Standardization & Cleaning**  
- **`standardize_trial_info(info)`** – Standardizes extracted trial details.  
- **`clean_response(response, max_length)`** – Cleans and truncates extracted text.  
- **`parse_llm_response(response)`** – Parses LLM output into structured trial data.  

## **Logging & Error Handling**  
All errors and processes are logged in `clinical_trials_processing.log`.  

## **Contributing**  
Feel free to submit issues and pull requests to improve the tool.  

## **License**  
This project is licensed under the **MIT License**.  
