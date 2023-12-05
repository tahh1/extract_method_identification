from sklearn.model_selection import train_test_split
import numpy as np, json, random, pickle, sys


def __get_data_from_jsonl(data_file):

    data, labels = [], []
    max_p, empty_repo = 0, 0
    # target = 202
    non_empty=0
    with open(data_file, 'r') as file:
        for i,line in enumerate(file):
            item = json.loads(line)
            if len(item['positive_case_methods'])==0:
                empty_repo+=1
                #target+=1
                continue 
            # if(i==target):
            #     break
            non_empty +=1
            if len(item['positive_case_methods'])>max_p:
                max_p=len(item['positive_case_methods'])

            # if len(data)>=10000:
            #     break            
            
            data+=item['positive_case_methods']

            labels.extend([1]*len(item['positive_case_methods']))
            data+=item['negative_case_methods']

            labels.extend([0]*len(item['positive_case_methods']))
        try:
            assert len(labels)==len(data)
        except AssertionError as e:
            print(len(labels))
            print(len(data))
    print("Processed repos", non_empty)
    print("Total samples - ", len(data))
    print("Maximum methods per case in a repo - ", max_p)
    print("Empty Repositories - ",empty_repo)
    return data, labels

def get_train_test_split(data, labels, test_ratio=0.2):
    train_data, test_data, train_label, test_label = train_test_split(data,labels, test_size=test_ratio, stratify=labels)
    print(f"Training sample length - {len(train_data)}. Test Sample length - {len(test_data)}")
    print(f"Training label length - {len(train_label)}. Test label length - {len(test_label)}")
    return train_data, test_data, train_label, test_label

def get_train_test_val_split(data, labels,val_ratio=0.2,test_ratio=0.1):
    train_val_data, test_data, train_val_label, test_label = train_test_split(data,labels, test_size=test_ratio, stratify=labels)

    train_data, val_data, train_label, val_label= train_test_split(data,labels, test_size=(val_ratio/(1-test_ratio)), stratify=labels)
    print(f"Training sample length - {len(train_data)}. Validation Sample length - {len(val_data)}- Test Sample length - {len(test_data)}")
    print(f"Training label length - {len(train_label)}. Validation label length - {len(val_label)}- Test label length - {len(test_label)}")
    return train_data, val_data, test_data ,train_label, val_label,test_label

def split_by_ratio(test_data, test_labels, ratio=0.85):

    combined_data = list(zip(test_data, test_labels))

    print("Test set splitting based on identified distribution....")
    ones_indices = [i for i, label in enumerate(test_labels) if label == 1]
    num_ones = len(ones_indices)
    num_to_remove = int(ratio * num_ones)

    random.shuffle(ones_indices)
    indices_to_remove = ones_indices[:num_to_remove]

    filtered_data = []
    filtered_labels = []

    for i, (data, label) in enumerate(combined_data):
        if i not in indices_to_remove or label == 0:
            filtered_data.append(data)
            filtered_labels.append(label)
    
    return filtered_data, filtered_labels

def save_nps(file_path, halfsize=True, split_by_size=True, rf= True, ae= True):

    data, labels = __get_data_from_jsonl(file_path)

    if(rf):
        print("--rf--")
        data_rf = data[len(data)//2:]
        labels_rf= labels[len(labels)//2:]
        
        rf_train_data, rf_test_data, rf_train_label,rf_test_label = get_train_test_split(data_rf, labels_rf,test_ratio=0.2)

        rf_train_data_np = np.asarray(rf_train_data)
        rf_train_label_np = np.asarray(rf_train_label)
        if split_by_size:
            rf_test_data,rf_test_label = split_by_ratio(rf_test_data, rf_test_label)
        rf_test_data_np = np.asarray(rf_test_data)
        rf_test_label_np = np.asarray(rf_test_label)


        print("rf Train Data Shape",rf_train_data_np.shape)
        print("rf Train Label Shape", rf_train_label_np.shape)


        
        print("rf Test Data Shape",rf_test_data_np.shape)
        print("rf Test Label Shape", rf_test_label_np.shape)
        print("rf test data ratio", len(np.nonzero(rf_test_label_np)[0]))
        with open ("../data/np_arrays/rf_train_data.npy","+wb") as f:
            np.save(f,rf_train_data_np)

        with open ("../data/np_arrays/rf_test_data.npy","+wb") as f:
            np.save(f,rf_test_data_np)

        with open ("../data/np_arrays/rf_train_label.npy","+wb") as f:
            np.save(f,rf_train_label_np)

        with open ("../data/np_arrays/rf_test_label.npy","+wb") as f:
            np.save(f,rf_test_label_np)

    
        del rf_train_data_np 
        del rf_train_label_np 

        del rf_test_data_np 
        del rf_test_label_np 
    if(ae):
        print("--ae--")
        data_ae = data[:len(data)//2]
        labels_ae = labels[:len(labels)//2]
        ae_train_data, ae_test_data,ae_train_label, ae_test_label=get_train_test_split(data_ae, labels_ae, test_ratio=0.3)
        ae_train_data_np = np.asarray(ae_train_data)
        ae_train_label_np = np.asarray(ae_train_label)
        # if split_by_size:
        #     ae_test_data,ae_test_label = split_by_ratio(ae_test_data, ae_test_label)
        ae_test_data_np = np.asarray(ae_test_data)
        ae_test_label_np = np.asarray(ae_test_label)
        print("ae Train Data Shape",ae_train_data_np.shape)
        print("ae Train Label Shape", ae_train_label_np.shape)

        print("ae test Data Shape",ae_test_data_np.shape)
        print("ae test Label Shape", ae_test_label_np.shape)

        with open ("../data/np_arrays/ae_train_data.npy","+wb") as f:
            np.save(f,ae_train_data_np)

        with open ("../data/np_arrays/ae_test_data.npy","+wb") as f:
            np.save(f,ae_test_data_np)

        with open ("../data/np_arrays/ae_train_label.npy","+wb") as f:
            np.save(f,ae_train_label_np)

        with open ("../data/np_arrays/ae_test_label.npy","+wb") as f:
            np.save(f,ae_test_label_np)
        del ae_train_data_np
        del ae_train_label_np
        del ae_test_data_np 
        del ae_test_label_np 



if __name__=="__main__":

    print("Start...")
    input_file_path = sys.argv[1]
    save_nps(f"../data/output/{input_file_path}",
             halfsize=False,
             split_by_size=True)