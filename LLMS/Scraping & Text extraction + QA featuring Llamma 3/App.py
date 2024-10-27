import requests
from bs4 import BeautifulSoup
import csv
import re
import logging
import time
from tqdm import tqdm
import os
from dotenv import load_dotenv
import sys
from datetime import datetime

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(filename='clinical_trials_processing.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def search_pubmed(keyword, page):
    """
    Search PubMed for clinical trials based on a keyword and page number.
    
    Args:
        keyword (str): The search term to use.
        page (int): The page number of results to retrieve.
    
    Returns:
        str: The HTML content of the search results page, or None if the request failed.
    """
    base_url = "https://pubmed.ncbi.nlm.nih.gov/"
    params = {
        "term": f"{keyword} AND Randomized Controlled Trial[Publication Type]",
        "filter": "simsearch1.fha",
        "page": page
    }
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        logging.error(f"Error fetching PubMed results: {str(e)}")
        return None

def get_total_pages(keyword):
    """
    Get the total number of pages for a given search keyword.
    
    Args:
        keyword (str): The search term to use.
    
    Returns:
        int: The total number of pages, or 0 if the request failed.
    """
    html_content = search_pubmed(keyword, 1)
    if html_content:
        soup = BeautifulSoup(html_content, 'html.parser')
        results_count = soup.find('span', class_='value').text.replace(',', '')
        return int(int(results_count) / 10) + 1
    return 0

def extract_trial_info(html_content):
    """
    Extract trial information from the HTML content of a PubMed search results page.
    
    Args:
        html_content (str): The HTML content of the search results page.
    
    Returns:
        list: A list of dictionaries containing trial titles and links.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    articles = soup.find_all('article', class_='full-docsum')
    trials = []
    for article in articles:
        title = article.find('a', class_='docsum-title').text.strip()
        link = "https://pubmed.ncbi.nlm.nih.gov" + article.find('a', class_='docsum-title')['href']
        trials.append({'title': title, 'link': link})
    return trials

def get_full_article_text(url):
    """
    Retrieve the full text of an article from its PubMed URL.
    
    Args:
        url (str): The URL of the article on PubMed.
    
    Returns:
        str: The abstract text of the article, or None if retrieval failed.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        abstract = soup.find('div', class_='abstract-content selected')
        if abstract:
            return abstract.get_text(strip=True)
        else:
            return "No abstract available"
    except requests.RequestException as e:
        logging.error(f"Error fetching full article text: {str(e)}")
        return None

def process_trial_with_llm(trial_text):
    """
    Process the trial text using a language model to extract structured information.

    Args:
        trial_text (str): The full text of the clinical trial article.

    Returns:
        str: The structured response from the language model, or None if processing failed.
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost",  # Replace with your actual app's URL
        "X-Title": "Clinical Trial Data Extractor"  # Replace with your app's name
    }
    prompt = f"""
    Analyze the following clinical trial information and provide a structured summary:

    {trial_text}

    Please provide the information in the following format:

    Trial Information:
    Trial_Info: [Brief description of the trial]
    NCT_Number: [NCT number if available, otherwise "NA"]
    Trial_Phase: [Trial phase if available, otherwise "NA"]
    Cancer_Type: [Type of cancer studied]
    Cancer_Description: [Brief description of the cancer type and stage]
    Trial_Sponsor: [Name of the trial sponsor]

    Study Groups:
    Group1: Description: [Brief description], Group_Type: [Control/Intervention], Drugs_Studied: [List of drugs], Treatment_ORR: [ORR if available], PFS: [PFS if available], OS: [OS if available], Discontinuation_Rate: [Rate if available], Endpoints_Met: [Yes/No/NA], Cancer_Stages: [Stages included], Targets: [Molecular targets if applicable], Previous_Drug_Types: [Types of previous treatments], Drug_Resistance: [Any information on drug resistance], Drug_Type_Resistance: [Specific drug types if resistant], Brain_Metastases: [Yes/No/NA], Previous_Surgery: [Yes/No/NA], Advanced_Cancer: [Yes/No/NA], Metastatic_Cancer: [Yes/No/NA], Previously_Untreated: [Yes/No/NA], Previous_Specific_Drugs: [List if applicable], Not_Previously_Taken_Drugs: [List if applicable], Therapy_Line: [First-line/Second-line/etc.], Treatment_Tolerance: [Any information on tolerance], Adverse_Reactions: [List of significant adverse reactions], Intervention_Drug_Approval: [Approval status if mentioned], Other_Efficacy_Data: [Any other relevant efficacy information]

    [Repeat Group information for each study group]

    Trial Results:
    Novel_Findings: [Brief description of novel findings]
    Conclusions: [Main conclusions of the trial]
    Unique_Information: [Any unique aspects of this trial]
    Subgroups_with_Heightened_Response: [Any subgroups that showed better response, if applicable]

    Please fill in the information as accurately as possible based on the provided text. If information for a field is not available, please use "NA".
    """
    data = {
        "model": "meta-llama/llama-3.1-70b-instruct",  # You can change this to another model if needed
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that analyzes clinical trial data."},
            {"role": "user", "content": prompt}
        ]
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        logging.error(f"Error in LLM processing: {e}")
        return None

def standardize_trial_info(info):
    """
    Standardize the trial information string.
    
    Args:
        info (str): The raw trial information string.
    
    Returns:
        str: A standardized string in the format "ID:Name:Patients".
    """
    id_match = re.search(r'(NCT\d+|ISRCTN\d+|ACTRN\d+)', info)
    trial_id = id_match.group(1) if id_match else "NoID"
    
    name_match = re.search(r':(.*?)(:|\(|$)', info)
    name = name_match.group(1).strip() if name_match else "Unnamed Trial"
    
    patients_match = re.search(r':(\d+)', info)
    patients = patients_match.group(1) if patients_match else "0"
    
    return f"{trial_id}:{name}:{patients}"

def clean_response(response, max_length=100):
    """
    Clean and truncate a response string.
    
    Args:
        response (str): The raw response string.
        max_length (int): The maximum length of the cleaned response.
    
    Returns:
        str: The cleaned and truncated response string.
    """
    if not response or response.strip().lower() in ['na', 'n/a', 'not specified', 'not applicable', 'not available', 'unknown']:
        return "NA"
    response = re.sub(r'\s*\([^)]*\)', '', response)  # Remove parenthetical explanations
    response = response.strip()
    if len(response) > max_length:
        return response[:max_length] + "..."
    return response

def parse_group_info(group_info):
    """
    Parse the group information string into type, intervention, and description.
    
    Args:
        group_info (str): The raw group information string.
    
    Returns:
        tuple: A tuple containing group_type, intervention, and description.
    """
    parts = group_info.split(':')
    group_type = clean_response(parts[0], max_length=30) if len(parts) > 0 else "NA"
    intervention = clean_response(parts[1], max_length=50) if len(parts) > 1 else "NA"
    description = clean_response(':'.join(parts[2:]), max_length=100) if len(parts) > 2 else "NA"
    
    # Remove redundancy between group_type and intervention
    if intervention.lower() in group_type.lower():
        group_type = "Intervention" if "intervention" in group_type.lower() else "Control"
    
    return group_type, intervention, description

def parse_llm_response(response):
    trials = []
    current_trial = {}
    
    # Define all expected fields
    trial_fields = [
        "Trial_Info", "NCT_Number", "Trial_Phase", "Cancer_Type", "Cancer_Description", "Trial_Sponsor",
        "Novel_Findings", "Conclusions", "Unique_Information", "Subgroups_with_Heightened_Response"
    ]
    
    group_fields = [
        "Description", "Group_Type", "Drugs_Studied", "Treatment_ORR", "PFS", "OS", "Discontinuation_Rate",
        "Endpoints_Met", "Cancer_Stages", "Targets", "Previous_Drug_Types", "Drug_Resistance",
        "Drug_Type_Resistance", "Brain_Metastases", "Previous_Surgery", "Advanced_Cancer",
        "Metastatic_Cancer", "Previously_Untreated", "Previous_Specific_Drugs",
        "Not_Previously_Taken_Drugs", "Therapy_Line", "Treatment_Tolerance",
        "Adverse_Reactions", "Intervention_Drug_Approval", "Other_Efficacy_Data"
    ]
    
    # Initialize the current trial with all fields
    for field in trial_fields:
        current_trial[field] = "NA"
    
    lines = response.split('\n')
    section = None
    current_group = None
    group_count = 0
    
    for line in lines:
        line = line.strip()
        
        if line.startswith("Trial Information:"):
            section = "trial_info"
        elif line.startswith("Study Groups:"):
            section = "groups"
        elif line.startswith("Trial Results:"):
            section = "results"
        
        if section == "trial_info":
            for field in trial_fields:
                if line.startswith(f"{field}:"):
                    current_trial[field] = line.split(":", 1)[1].strip()
        
        elif section == "groups":
            match = re.match(r'Group(\d+):(.*)', line)
            if match:
                group_num = int(match.group(1))
                group_count = max(group_count, group_num)
                current_group = group_num
                group_info = match.group(2).strip()
                for field in group_fields:
                    if f"{field}:" in group_info:
                        value = group_info.split(f"{field}:", 1)[1].split(",", 1)[0].strip()
                        current_trial[f"Group{group_num}_{field}"] = value
        
        elif section == "results":
            if line.startswith("Novel Findings:"):
                current_trial["Novel_Findings"] = line.split(":", 1)[1].strip()
            elif line.startswith("Conclusions:"):
                current_trial["Conclusions"] = line.split(":", 1)[1].strip()
            elif line.startswith("Unique Information:"):
                current_trial["Unique_Information"] = line.split(":", 1)[1].strip()
            elif line.startswith("Subgroups with Heightened Response:"):
                current_trial["Subgroups_with_Heightened_Response"] = line.split(":", 1)[1].strip()
    
    # Ensure all group fields are present for each group
    for i in range(1, group_count + 1):
        for field in group_fields:
            if f"Group{i}_{field}" not in current_trial:
                current_trial[f"Group{i}_{field}"] = "NA"
    
    trials.append(current_trial)
    return trials

def save_to_csv(trials, filename):
    # Find the maximum number of groups across all trials
    max_groups = 0
    for trial in trials:
        group_fields = [key for key in trial.keys() if key.startswith("Group")]
        if group_fields:
            max_group = max([int(key.split("_")[0][5:]) for key in group_fields])
            max_groups = max(max_groups, max_group)
    
    fieldnames = [
        "Trial_Info", "NCT_Number", "Trial_Phase", "Cancer_Type", "Cancer_Description", "Trial_Sponsor",
        "Novel_Findings", "Conclusions", "Unique_Information", "Subgroups_with_Heightened_Response"
    ]
    
    group_fields = [
        "Description", "Group_Type", "Drugs_Studied", "Treatment_ORR", "PFS", "OS", "Discontinuation_Rate",
        "Endpoints_Met", "Cancer_Stages", "Targets", "Previous_Drug_Types", "Drug_Resistance",
        "Drug_Type_Resistance", "Brain_Metastases", "Previous_Surgery", "Advanced_Cancer",
        "Metastatic_Cancer", "Previously_Untreated", "Previous_Specific_Drugs",
        "Not_Previously_Taken_Drugs", "Therapy_Line", "Treatment_Tolerance",
        "Adverse_Reactions", "Intervention_Drug_Approval", "Other_Efficacy_Data"
    ]
    
    for i in range(1, max_groups + 1):
        for field in group_fields:
            fieldnames.append(f"Group{i}_{field}")
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for trial in trials:
            # Ensure all fields are present in the trial dictionary
            for field in fieldnames:
                if field not in trial:
                    trial[field] = "NA"
            writer.writerow(trial)

def main():
    """
    Main function to orchestrate the PubMed clinical trial data extraction process.
    This function handles user input, initiates the search process, and manages the
    overall flow of data extraction and processing.
    """
    
    keyword = input("Enter the keyword to search in PubMed: ")
    total_pages = get_total_pages(keyword)
    print(f"Total number of pages: {total_pages}")
    start_page = int(input("Enter the starting page number: "))
    end_page = int(input("Enter the ending page number: "))

    all_trials = []
    print("Processing trials...")

    with tqdm(total=end_page - start_page + 1, desc="Pages processed", unit="page") as pbar:
        for page in range(start_page, end_page + 1):
            logging.info(f"Processing page {page}")
            html_content = search_pubmed(keyword, page)
            if html_content is None:
                continue
            trials = extract_trial_info(html_content)

            for trial in trials:
                try:
                    print(f"Processing: {trial['title']}")
                    logging.info(f"Processing: {trial['title']}")
                    trial_text = get_full_article_text(trial['link'])
                    if trial_text is None:
                        continue
                    llm_response = process_trial_with_llm(trial_text)
                    if llm_response:
                        parsed_trials = parse_llm_response(llm_response)
                        all_trials.extend(parsed_trials)
                    else:
                        logging.warning(f"No LLM response for trial: {trial['title']}")
                except Exception as e:
                    logging.error(f"Error processing trial {trial['title']}: {str(e)}")

                time.sleep(1)

            pbar.update(1)

    print("Processing complete.")

    if all_trials:
        save_to_csv(all_trials, 'clinical_trials_data.csv')
        print(f"Data has been saved to clinical_trials_data.csv. Total trials processed: {len(all_trials)}")
    else:
        print("No data was successfully processed and parsed.")

if __name__ == "__main__":
    main()
