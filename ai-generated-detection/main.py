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

def evalaute_mode(path, threshold):

    ai_files = list_files(path)

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
        if f_score > threshold:
            fake_correct += 1

    t_p = float((fake_correct * 100)) / total_count
    print(f"Total : {total_count}")
    print(f"Classified correctly: {fake_correct}")
    print(f"True positive % is {t_p}%")

def process_ai_gen_image(path):
    ai_score = 0
    items = pipe(path)
    for item in items:
            
            # if item['label'] == 'artificial':
            if item['label'] == 'FAKE':
            
                ai_score = item['score']
                break
    return ai_score