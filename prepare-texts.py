#!/usr/bin/python

import logging
import sys
import os
import string

log = logging.getLogger()
log.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
log.addHandler(handler)

'''
Simple text files preparation. Merges raw text files into a single one with the format of one document per line.
Converts everything to lower case and removes punctuation. 

Usage: python prepare_texts.py <directory_to_process> <result_file_name>
'''

# Translator for stripping punctuation. Not stripping actually but replacing with space to prevent
# merging words into one. Anyway all multiple spaces will be normalized later.
strip_table = str.maketrans({key: ' ' for key in string.punctuation})


def process_dir(dir_in, file_out):
    log.info("Processing directory %s ", dir_in)
    log.info("Result file %s ", file_out)
    i = 0
    with open(file_out, mode='w', encoding='utf8') as f_out:
        for i, f_name in enumerate(os.listdir(dir_in)):
            f_path = os.path.join(dir_in, f_name)
            if os.path.isfile(f_path) and f_name.endswith(".txt"):
                    if i % 1000 == 0:
                        log.info(str(i) + ":" + f_path)
                    f_out.write(process_file(f_path))

        log.info("*** Written %i documents, %i bytes", i+1, f_out.tell())


def process_file(file_path):
    # process the text to one string etc
    with open(file_path, encoding="utf8") as f:
        txt = f.read()\
            .replace('<br />', ' ') \
            .lower() \
            .translate(strip_table)  # removing punctuation
        return ' '.join(txt.split()) + '\n'  # removing multiple spaces, tabs and CRLFs


def_dir_in = './aclImdb/train/unsup/'
def_file_out = './train-unsup.txt'

if __name__ == '__main__':
    if len(sys.argv) > 2:
        process_dir(sys.argv[1], sys.argv[2])
    else:
        process_dir(def_dir_in, def_file_out)
