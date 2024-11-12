# Resume Parser

This is a simple tool parses PDF resumes, extracting key information such as personal details, work experience, education, and skills.

## Features

* PDF text extraction using PyPDF2
* Named entity recognition for name and organization extraction using spaCy
* Contact information extraction using regex patterns
* Skills matching against a predefined list
* Education and work experience parsing

## Usage

1. Install required libraries: `pip install spacy PyPDF2`
2. Download spaCy model: `python -m spacy download en_core_web_sm`
3. Run the script: `python resume_parser.py path/to/resume.pdf`

## Output

The script outputs a dictionary containing the following information:

* Full name
* Contact information (email, phone, LinkedIn, GitHub, Medium)
* Summary or objective statement
* List of identified skills
* Work experience (company, job title, dates)
* Education details (degree, institution, dates)
* Certifications
* Projects# resumer-parser
# resumer-parser
