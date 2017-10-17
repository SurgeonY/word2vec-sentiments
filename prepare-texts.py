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
Simple text files preparation. Merges raw movie review files into single one with the format 
removing all punctuation and de-capitalizing. Converts everything to lower case and removing punctuation 
'''

dir_in = './aclImdb/train/unsup/'
file_out = './train-unsup.txt'

def __main__():
    strip_table = str.maketrans({key: ' ' for key in string.punctuation})

    with open(file_out, mode='w', encoding='utf8') as f_out:
        for i, f_name in enumerate(os.listdir(dir_in)):
            if f_name.endswith(".txt"):
                f_path = os.path.join(dir_in, f_name)
                log.info(str(i) + ":" + f_path)

                # process the text to one string etc
                txt = read_file(f_path)\
                    .replace('<br />', ' ')\
                    .lower()\
                    .translate(strip_table)   # removing punctuation
                f_out.write(' '.join(txt.split()) + '\n')  # removing multiple spaces

        log.info("*** Written %i bytes", f_out.tell())


def read_file(file_path):
    with open(file_path, encoding="utf8") as f:
        return f.read()


__main__()
