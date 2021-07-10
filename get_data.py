from google_drive_downloader import GoogleDriveDownloader as gdd
import os
files = {
    "my_module_1400_02_10.pt": "126sTLOrOzBrq6NdwFk9__nkHqDoHx3O7",
    "NER_model.pt": "1a4Mt49vssYO9PlZyzNOE1MKdqgCNrv-M",
    "peyma_ner_id2tag.pickle": "1-3FUUQC2gsh3eJUjEJF3qwgemy2YMo_8",
    "peyma_ner_tag2id.pickle": "1--oMUwhJg4F_H3HlG8mYxD5zhSnWf45t",
}

for file_name, gdrive_id in files.items():
    if not os.path.exists("./data/"+file_name):
        print("Download "+file_name)
        gdd.download_file_from_google_drive(file_id=gdrive_id,
                                            dest_path='./data/' + file_name+"_cache")
        os.rename('./data/' + file_name+"_cache", './data/' + file_name)

print("Get Data finished!")
