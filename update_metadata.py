import os
import json



def update_metadata(model_data,Whole_model=0):
    if(Whole_model==1):
        Model_History = {}
        if os.path.exists("Model_History.json"):
            with open("Model_History.json", 'r') as f:
                Model_History = json.load(f)
        
        Model_History[model_data['version']] = {
        'version_description': model_data['version_description'],
        'Selected_Features':model_data['Selected_Features'],
        'num_rows_train':model_data['num_rows_train'],
        'num_rows_val': model_data['num_rows_val'],
        'model':'Ridge',
        'data_path_train':model_data['data_path_train'],
        'data_path_val':model_data['data_path_val'],
        'data_path_test':model_data['data_path_test'],
        'train_r2':model_data['train_r2'],
        'val_r2':model_data['val_r2'],
        'timestamp': model_data['timestamp']
    }

        # Save updated history
        with open("Model_History.json", 'w') as f:
            json.dump(Model_History, f, indent=4)
    else:
        Base_Line_Model_History = {}
        if os.path.exists("Base_Line_Model_History.json"):
            with open("Base_Line_Model_History.json", 'r') as f:
                Base_Line_Model_History = json.load(f)
        
        Base_Line_Model_History[model_data['version']] = {
        'version_description': model_data['version_description'],
        'Selected_Features':model_data['Selected_Features'],
        'num_rows_train':model_data['num_rows_train'],
        'num_rows_val': model_data['num_rows_val'],
        'model':'Ridge',
        'data_path_train':model_data['data_path_train'],
        'data_path_val':model_data['data_path_val'],
        'data_path_test':model_data['data_path_test'],
        'train_r2':model_data['train_r2'],
        'val_r2':model_data['val_r2'],
        'test_r2':model_data['test_r2'],
        'timestamp': model_data['timestamp']
    }

        # Save updated history
        with open("Base_Line_Model_History.json", 'w') as f:
            json.dump(Base_Line_Model_History, f, indent=4)


