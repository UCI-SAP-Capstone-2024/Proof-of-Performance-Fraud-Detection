from ConfidenceScore import calculate_final_confidence_score

def decide_action_based_on_score(image, dealSheetPath):
    score = calculate_final_confidence_score(image, dealSheetPath)
    
    if score == -1:
        return "Reject: The image is already used"
    elif score < 30:
        return "Reject: High probability of fraud"
    elif score > 70:
        return "Auto-Pass: No apparent signs of fraud"
    else:
        return "Manual Review: Potential concerns detected"

# def main():
#     image = "./Data/DocumentImages/Image-0.png"
#     dealSheetPath  = "./Data/DealSheets/RED BULL CHERRY FLAVOR LAUNCH.docx"
#     decision = decide_action_based_on_score(image, dealSheetPath)
#     print(decision)

def main(dealSheetPath, image):
    decision = decide_action_based_on_score(image, dealSheetPath)
    print(decision)
    return decision

# if __name__ == "__main__":
#     main()
