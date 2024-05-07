from PIL import Image
import imagehash
from DuplicateImageDetection import check_image_exists
from DealSheetExtraction import process_file

def check_image_manipulation(image):
    return 1

def check_ai_generated_image(image):
    return 1

def is_image_already_used(image):
    return check_image_exists(image)

def are_store_details_correct(image):
    return 1

def are_details_valid_in_deal_sheet(dealSheetPath):
    deal_sheet_contents = process_file(dealSheetPath)
    if deal_sheet_contents["Deal Start Date"]=='January 01, 2024' and deal_sheet_contents["Deal End Date"]=='March 01, 2024':
        return 1
    return 0

def calculate_final_confidence_score(image, dealSheetPath):
    if is_image_already_used(image):
        return -1
    
    scores = {
        'deal_sheet': are_details_valid_in_deal_sheet(dealSheetPath) * 0.50,
        'image_manipulation': check_image_manipulation(image) * 0.25,
        'ai_generated_image': check_ai_generated_image(image) * 0.15,
        'store_details': are_store_details_correct(image) * 0.10,
    }
    
    final_score = sum(scores.values()) * 100
    return final_score
