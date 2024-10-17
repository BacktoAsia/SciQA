import json
import shutil
from scripts.convert_sqa_to_llava import convert_to_llava

# dir of the raw data
base_dir = '/Users/shuogudaojin/Public/Database/ScienceQA/raw_data'
prompt =  "QCM-LEA"
splits = ['train', 'val', 'minival', 'test', 'minitest']

#---------------- Convert json to chat form -----------
for split in splits:
    convert_to_llava(base_dir=base_dir, split=split)
    
#---------------- Merge the dataset except 'test' together -------------
train_path = base_dir + '/llava_train_QCM-LEA.json'
test_path = base_dir + '/llava_test_QCM-LEA.json'
destination = '/Users/shuogudaojin/Public/Database/ScienceQA/chat_json'
merged_split = ['val', 'minitest', 'minival']

with open(train_path, 'r') as file:
    train = json.load(file)
with open(test_path, 'r') as file:
    test = json.load(file)
    
# Copy train dataset to destination
shutil.copy(train_path, destination)
shutil.copy(test_path, destination)
train_path = destination + '/llava_train_QCM-LEA.json'
test_path = destination + '/llava_test_QCM-LEA.json'

for split in merged_split:
    path = base_dir + '/llava_' + split + '_QCM-LEA.json'
    
    with open(path, 'r') as file:
        val = json.load(file)
         
    train = train + val
    
# Remove part of test data to train
remove_len = len(test) // 2
for i in range(remove_len):
    train.append(test[i])
    del test[i]
        
# --------------- Write the result into json ---------------
with open(train_path, 'w') as file:
    json.dump(train, file)

with open(test_path, 'w') as file:
    json.dump(test, file)


