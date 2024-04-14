import os
import json

def extract_info(directory):
    data = []
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".wav") and file != "merge_result.wav":  # check
                filepath = os.path.join(subdir, file)
                filepath = filepath.replace("\\", "/")  # replace
                filename = os.path.splitext(file)[0]  # remove file_ex
                file_info = filename.split('_')  # Split filename by underscore
                subdir = subdir.replace("\\", "/")  # replace
                # print(subdir.split("/"))
                dialect_region = subdir.split("/")[-2]  # extract
                gender = subdir.split("/")[-1][0]
                # print(file_info)
                # speaker_id = file_info[0][1:]  # test
                speaker_id = subdir.split("/")[-1][1:]
                sentence_type = file_info[0][:2]  # extract prefix
                sentence_id = file_info[0][2:]  # extract suffix
                
                data.append({
                    "filepath": filepath,
                    "dialect_region": dialect_region,
                    "gender": gender,
                    "speaker_id": speaker_id,
                    "sentence_type": sentence_type,
                    "sentence_id": sentence_id
                })
    return data

def main():
    train_data = extract_info("./timit_lite/TRAIN")
    test_data = extract_info("./timit_lite/TEST")

    with open("train_info.json", "w") as train_file:
        json.dump(train_data, train_file, indent=4)

    with open("test_info.json", "w") as test_file:
        json.dump(test_data, test_file, indent=4)

if __name__ == "__main__":
    main()
