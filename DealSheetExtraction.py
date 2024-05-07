import os
import re
import queue
from dateutil.parser import *
from spire.doc import *
from spire.doc.common import *
import fitz
import docx
import spacy
import redis
from PIL import Image
import imagehash
import io
from docx.parts.image import ImagePart

nlp = spacy.load('en_core_web_md')


def extract_text_from_pdf(file):
    text = ""
    with fitz.open(file) as pdf_file:
        for page_num in range(len(pdf_file)):
            text += pdf_file[page_num].get_text()
    return text


def extract_text_from_docx(file):
    # doc = docx.Document(file)
    text = ""
    for paragraph in file.paragraphs:
        text += paragraph.text + "\n"
    return text


def clean_text(text):
    text = text.replace("\r\n", " ")
    text = re.sub(" +", " ", text)
    return text


def clean_text_from_pdf(file):
    text = extract_text_from_pdf(file)
    return clean_text(text)


def clean_text_from_docx(file):
    text = extract_text_from_docx(file)
    return clean_text(text)


def get_invoice_number(file):
    if file.endswith('.pdf'):
        text = extract_text_from_pdf(file)
    elif file.endswith('.docx'):
        text = extract_text_from_docx(file)
    else:
        raise ValueError("Unsupported file format")

    text = text.replace("\r\n", "|")
    text = re.sub(" +", " ", text)

    tokens = [i for i in text.split('|') if len(i) > 1]
    text = '|'.join(tokens)

    text = text.lower()
    text = re.sub(" number:| no\.| n\.", " number", text)

    valid_matches = []
    keywords = ["invoice number", "lading number", "invoice", "lading", "number"]
    for keyword in keywords:
        token = extract_tag_data(keyword, text, 'reverse')
        if len(token) > 0:
            valid_matches.append(token)

    valid_matches = [i.upper() for i in valid_matches]

    if len(valid_matches) > 0:
        invoice_number = valid_matches[0]
        if len(invoice_number) > 20:
            invoice_number = invoice_number.split('/')[0]
    else:
        invoice_number = "No Invoice Number found"

    return invoice_number


def extract_tag_data(keyword, text, direction):
    if direction == 'forward':
        regex = re.compile(f"{keyword}(.+?)\|", re.IGNORECASE)
        matches = regex.findall(text)
        if matches:
            return matches[-1].strip()
    elif direction == 'reverse':
        regex = re.compile(r"\|(.+?)" + f"{keyword}", re.IGNORECASE)
        matches = regex.findall(text)
        if matches:
            return matches[0].strip()
    return ""


def get_deal_dates(text):
    possible_dates = []
    
    doc = nlp(text)
    possible_dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]

    start_date = None
    end_date = None

    for date in possible_dates:
        try:
            parsed_date = parse(date)
            if start_date is None or parsed_date < start_date:
                start_date = parsed_date
            if end_date is None or parsed_date > end_date:
                end_date = parsed_date
        except:
            continue

    if start_date and end_date:
        return start_date.strftime("%B %d, %Y"), end_date.strftime("%B %d, %Y")
    else:
        return "No Start and End Dates Found"


def extract_collaborators(text):
    collaborators = {}
    lines = text.split('\n')

    for line_index, line in enumerate(lines):
        if "Name:" in line:
            name_index = line.index("Name:")
            name = line[name_index + len("Name:"):].strip()
            if name:
                if line_index < len(lines) - 1:
                    address = lines[line_index + 1].strip()
                    collaborators[name] = address

    return collaborators


def extract_integrations(file_text):
    integrations = {}

    integration_types = ["In-Store Integrations", "Car Display Integrations", "Online Integrations"]

    for integration_type in integration_types:
        start_index = file_text.find(integration_type)
        if start_index != -1:
            next_integration_index = len(file_text)
            for next_type in integration_types:
                if next_type != integration_type:
                    next_start_index = file_text.find(next_type, start_index + len(integration_type))
                    if next_start_index != -1 and next_start_index < next_integration_index:
                        next_integration_index = next_start_index
            integrations[integration_type] = file_text[start_index:next_integration_index].strip()

    return integrations


def image_to_hex(image_data):
    image = Image.open(io.BytesIO(image_data))
    return str(imagehash.average_hash(image))

def extract_images_from_docx(input_file,  redis_host = 'localhost', redis_port = 6379, redis_db = 0):
    r = redis.Redis(host=redis_host, port=redis_port, db=redis_db)
    
    images = []

    for rel in input_file.part.rels.values():
        if "image" in rel.reltype:
            target_part = rel.target_part
            if isinstance(target_part, ImagePart):
                image_bytes = target_part.blob
                images.append(image_bytes)
    
    for i, image_data in enumerate(images):
        file_name = f"Image-{i}.png"
        image_hash = image_to_hex(image_data)
        r.set(image_hash, file_name)


def process_pdf(file):
    file_text = clean_text_from_pdf(file)
    deal_dates = get_deal_dates(file_text)
    collaborators = extract_collaborators(file_text)
    integrations = extract_integrations(file_text)
    return {
        "Deal Start Date": deal_dates[0],
        "Deal End Date": deal_dates[1],
        "Collaborators": collaborators,
        "Integrations": integrations
    }


def process_docx(file, redis_host = 'localhost', redis_port = 6379, redis_db = 0):
    file_text = clean_text_from_docx(file)
    deal_dates = get_deal_dates(file_text)
    collaborators = extract_collaborators(file_text)
    integrations = extract_integrations(file_text)

    extract_images_from_docx(file, redis_host, redis_port, redis_db)
    
    return {
        "Deal Start Date": deal_dates[0],
        "Deal End Date": deal_dates[1],
        "Collaborators": collaborators
    }


def process_file(file):
    return process_docx(file)
    # if file.endswith('.pdf'):
    #     return process_pdf(file)
    # elif file.endswith('.docx'):
    #     return process_docx(file)
    # else:
    #     raise ValueError("Unsupported file format")
