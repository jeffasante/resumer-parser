import spacy
import PyPDF2
import pandas as pd
import re
from typing import Dict, Any, List, Union, Optional
from spacy.matcher import Matcher
import logging
import time
import argparse

# Logging configuration
logging.basicConfig(level=logging.INFO)

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF file."""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        return " ".join(page.extract_text() for page in reader.pages if page.extract_text())


def extract_full_name(doc: spacy.tokens.Doc) -> str:
    """Extract full name using named entity recognition and remove any email."""
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            # Remove email addresses from the extracted name
            name_without_email = re.sub(r'\S+@\S+', '', ent.text)
            # Remove any extra whitespace
            return ' '.join(name_without_email.split())
    return "Unknown"


def extract_contact_info(text: str) -> Dict[str, str]:
    """Extract contact information using regex."""
    contact_info = {}

    # Extract email
    email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    if email_match:
        contact_info["email"] = email_match.group()

    # Extract phone
    phone_match = re.search(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text)
    if phone_match:
        contact_info["phone"] = phone_match.group()

    # Extract LinkedIn
    linkedin_match = re.search(r'linkedin\.com/in/[\w-]+', text)
    if linkedin_match:
        contact_info["linkedin"] = linkedin_match.group()

    # Extract Medium
    medium_match = re.search(r'medium\.com/@[\w.-]+', text, re.IGNORECASE)
    if medium_match:
        contact_info["medium"] = medium_match.group()
    
    # Extract GitHub
    github_match = re.search(r'(?:github\.com/|github: )[\w-]+', text, re.IGNORECASE)
    if github_match:
        contact_info["github"] = github_match.group().split('/')[-1].strip()

    return contact_info


def extract_summary(doc: spacy.tokens.Doc) -> str:
    """
    Extract the summary section from the resume text.

    Args:
    text (str): The resume text.

    Returns:
    str: Extracted summary or an empty string if not found.
    """

    # Keywords to identify a summary section
    summary_keywords = ["summary", "objective", "profile", "about me",]

    # Look through the sentences
    summary = []
    in_summary = False

    for sent in doc.sents:
        clean_sent = sent.text.strip().lower()
        # print(clean_sent)
        # If the sentence contains any of the summary-related keywords, start capturing the sentences
        if any(keyword in clean_sent for keyword in summary_keywords):
            in_summary = True
            summary.append(sent.text.strip())
            continue
        
        # If we're in the summary section and hit a potential section header, stop capturing
        if in_summary:
            if "experience" in clean_sent:
                break  # Stop if we hit the next section
            else:
                summary.append(sent.text.strip())

    # Return the extracted summary or an empty string
    return " ".join(summary).strip()


def extract_skills(doc: spacy.tokens.Doc, skill_list: List[str]) -> List[str]:
    """Extract skills using keyword matching."""
    tokens = [token.text.lower() for token in doc if not token.is_stop]
    noun_chunks = [chunk.text.lower() for chunk in doc.noun_chunks]
    
    skills = []
    for skill in skill_list:
        if skill.lower() in tokens or skill.lower() in noun_chunks:
            skills.append(skill.capitalize())
    
    return list(set(skills))  # Remove duplicates and preserve case


def load_skills_from_csv(file_path: str) -> List[str]:
    """Load skills from a CSV file."""
    data = pd.read_csv(file_path)
    return list(data.columns.values)

def extract_work_experience(doc: spacy.tokens.Doc) -> List[Dict[str, Any]]:
    """Extract work experience information."""
    experience = []
    current_job = {}

    for ent in doc.ents:
        if ent.label_ == "ORG" and not current_job.get("company"):
            current_job["company"] = ent.text
        elif ent.label_ == "DATE" and not current_job.get("dates"):
            current_job["dates"] = ent.text

        if len(current_job) == 2:
            # Assume the next sentence is the job title
            for sent in doc.sents:
                if sent.start > ent.end:
                    current_job["job_title"] = sent.text.strip()
                    break

            experience.append(current_job)
            current_job = {}

    return experience


def extract_education(input_data: Union[spacy.tokens.Doc, str]) -> List[Dict[str, Any]]:
    """
    Extract education information from either a spaCy Doc object or a plain text string.

    Args:
    input_data (Union[spacy.tokens.Doc, str]): The input data, either a spaCy Doc or a string.

    Returns:
    List[Dict[str, Any]]: A list of dictionaries containing education information.
    """
    education = []
    current_edu = {}
    
    # Determine input type and convert to lines
    if isinstance(input_data, spacy.tokens.Doc):
        lines = [token.text for token in input_data if not token.is_space]
    elif isinstance(input_data, str):
        lines = [line.strip() for line in input_data.split('\n') if line.strip()]
    else:
        raise ValueError("Input must be either a spaCy Doc object or a string")
    
    # Keywords to identify the education section
    section_keywords = ["education", "academic background", "academic history"]
    
    # Flag to track if we're in the education section
    in_section = False
    
    for i, line in enumerate(lines):
        clean_line = line.lower()
        
        # Check for section header
        if any(keyword in clean_line for keyword in section_keywords):
            in_section = True
            continue
        
        # Check for end of section
        if in_section and any(keyword in clean_line for keyword in ["experience", "skills", "projects"]):
            in_section = False
            if current_edu:
                education.append(current_edu)
                current_edu = {}
            continue
        
        # If we're in the section, process it
        if in_section:
            # Try to identify degree
            degree_match = re.match(r'(bachelor|master|phd|doctorate|bs|ba|ms|ma|mba|md|jd|b\.s\.|m\.s\.)', clean_line, re.IGNORECASE)
            if degree_match:
                if current_edu:
                    education.append(current_edu)
                    current_edu = {}
                current_edu['degree'] = line + " " + lines[i+1] if i+1 < len(lines) else line
                continue
            
            # Try to identify institution
            if 'institution' not in current_edu and 'degree' in current_edu:
                current_edu['institution'] = line
                continue
            
            # Try to identify location
            if 'location' not in current_edu and 'institution' in current_edu:
                current_edu['location'] = line
                continue
            
            # Try to identify dates
            date_match = re.search(r'\d{4}\s*-\s*\d{4}|\d{4}\s*-\s*(present|current)', line, re.IGNORECASE)
            if date_match and 'dates' not in current_edu:
                current_edu['dates'] = date_match.group()
                continue
            
            # If we can't categorize the line, add it to additional_info
            if 'additional_info' not in current_edu:
                current_edu['additional_info'] = [line]
            else:
                current_edu['additional_info'].append(line)
    
    # Add any remaining education info
    if current_edu:
        education.append(current_edu)
    
    return education

def extract_awards_and_certifications(text: str) -> List[str]:
    """Extract awards and certifications from the text as a single category."""
    awards_and_certs = []
    
    # Split the text into lines for easier parsing
    lines = text.split('\n')
    
    # Keywords to identify the awards and certifications section
    section_keywords = ["award", "certification", "certificate", "achievement"]
    
    # Flag to track if we're in the awards and certifications section
    in_section = False
    
    for line in lines:
        clean_line = line.strip().lower()
        
        # Check for section header
        if any(keyword in clean_line for keyword in section_keywords):
            in_section = True
            continue
        
        # Check for end of section (usually when we hit another major section)
        if in_section and clean_line and clean_line[0].isupper() and clean_line.endswith(':'):
            in_section = False
        
        # If we're in the section and the line is not empty, add it to the list
        if in_section and clean_line:
            # Remove any date patterns (assuming dates are in parentheses)
            line_without_date = re.sub(r'\([^)]*\)', '', line).strip()
            if line_without_date:
                awards_and_certs.append(line_without_date)
    
    return awards_and_certs


def extract_section(text: str, section_header: str, stop_headers: List[str]) -> List[str]:
    """Extract lines belonging to a specific section until the next unrelated section."""
    lines = text.split('\n')
    section_content = []
    in_section = False

    for line in lines:
        clean_line = line.strip()

        # Detect the section header
        if section_header.lower() in clean_line.lower():
            in_section = True
            continue

        # Stop if we hit a stop header
        if in_section and any(stop_header.lower() in clean_line.lower() for stop_header in stop_headers):
            break

        # If in the correct section, gather the content
        if in_section:
            section_content.append(clean_line)

    # Remove empty lines and return the content
    return [line for line in section_content if line]

def extract_publications_and_conferences(text: str) -> List[str]:
    """Extract the Publications & Conferences section."""
    stop_headers = ["projects", "education", "work experience", "skills", "certifications", "awards", "interests"]
    return extract_section(text, "publications & conferences", stop_headers)

def extract_projects(text: str) -> List[str]:
    """Extract the Projects section."""
    stop_headers = ["publications", "education", "work experience", "skills", "certifications", "awards", "interests"]
    return extract_section(text, "projects", stop_headers)


def parse_resume_pdf(pdf_path: str, skill_file: str) -> Dict[str, Any]:
    """
    Parse a PDF resume and extract key information using spaCy and PyPDF2.

    Args:
    pdf_path (str): Path to the PDF file
    skill_file (str): Path to the skills CSV file

    Returns:
    Dict[str, Any]: Dictionary containing extracted information
    """
    # Load skills from CSV file
    skills_list = load_skills_from_csv(skill_file)

    # Initialize result dictionary
    result = {
        "full_name": "",
        "contact_info": {},
        "summary": "",
        "skills": [],
        "work_experience": [],
        "education": [],
        "certifications": [],
        "projects": [],
    }

    # Extract text from PDF
    text = extract_text_from_pdf(pdf_path)

    # Process text with spaCy
    doc = nlp(text)

    # Extract information
    result["full_name"] = extract_full_name(doc)
    result["contact_info"] = extract_contact_info(text)
    result["summary"] = extract_summary(doc)
    result["skills"] = extract_skills(doc, skills_list)
    result["work_experience"] = extract_work_experience(doc)
    result["education"] = extract_education(doc)
    result["certifications"] = extract_awards_and_certifications(text)
    result["projects"] = extract_projects(text)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse a PDF resume and extract key information.")
    parser.add_argument("pdf_path", help="Path to the PDF resume file")
    args = parser.parse_args()

    # Constants
    SKILL_FILE = 'skills.csv'
    SPACY_MODEL = 'en_core_web_sm'

    # Load spaCy model
    try:
        nlp = spacy.load(SPACY_MODEL)
    except OSError:
        logging.error(f"Download the '{SPACY_MODEL}' spaCy model using 'python -m spacy download {SPACY_MODEL}'")
        exit(1)

    start_time = time.time()
    resume_data = parse_resume_pdf(args.pdf_path, SKILL_FILE)
    end_time = time.time()
    logging.info(f"Parsing completed in {end_time - start_time:.2f} seconds")
    
    from pprint import pprint
    pprint(resume_data)
