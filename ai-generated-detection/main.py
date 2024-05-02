# Load model directly
# Use a pipeline as a high-level helper
# from transformers import pipeline



import os
from transformers import pipeline

def list_files(directory):
    files = []
    for file in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, file)):
            files.append(file)
    return files
pipe = pipeline("image-classification", model="dima806/ai_vs_real_image_detection")

# pipe = pipeline("image-classification", model="Organika/sdxl-detector")



ai_files = list_files('images/fake')

total_count = len(ai_files)
fake_correct = 0
for ai_file in ai_files[:]:
    f = 'images/fake/' + ai_file
    res = pipe(f)
    print(res)
    f_score = 0
    r_score = 1
    for item in res:
        
        # if item['label'] == 'artificial':
        if item['label'] == 'FAKE':
        
            f_score = item['score']
            break
       
    print(f_score)
    if f_score > 0.5:
        fake_correct += 1

t_p = float((fake_correct * 100)) / total_count
print(f"Total : {total_count}")
print(f"Classified correctly: {fake_correct}")
print(f"True positive % is {t_p}%")
        
    
        